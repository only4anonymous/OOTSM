#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import re
# LoRA 相关
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import sys
project_root = "your/project/path"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
#################################################
# 你已有的 AG 数据集 + 常量
# from dataloader.action_genome.ag_dataset import AG, cuda_collate_fn, const
#################################################
# 这里只保留REL_CLASSES用于关系数
REL_CLASSES = [
    'looking_at', 'not_looking_at', 'unsure',
    'above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in',
    'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on'
]

ATTN_REL_CLASSES = ['looking_at', 'not_looking_at', 'unsure']
SPAT_REL_CLASSES = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
CONT_REL_CLASSES = ['carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship', 'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']
NUM_REL_CLASSES = len(REL_CLASSES)

# _PATTERN_LINE = re.compile(
#     r'object:\s*([^P]+?)(?=\s*Person)\s*Person attention to [^:]+:\s*([^,]+),\s*[^:]+?\s*located relative to person:\s*([^,]+),\s*Person contact with [^:]+:\s*([^,\.]+)',
#     re.IGNORECASE
# )
_PATTERN_LINE = re.compile(
    r'(?:Frame \d+\.\.\d+: |Frame \d+: )?object:\s*([^P]+?)(?=\s*attention:)\s*attention:\s*([^,]+),\s*spatial:\s*([^,]+),\s*contact:\s*([^,\.]+)',
    re.IGNORECASE
)

#################################################
# 1. 自定义数据集：AGForLLM (行级别对齐)
#################################################
class AGForLLM(Dataset):
    """
    将 Action Genome 数据集转换为: 
      - prompt_text: 观测段（合并后的文本）
      - target_text: 未来段（合并后的文本）
      - line_texts: 将 target_text 按“每个合并段”拆分成多行，每行为一个描述文本（不依赖 <obj> 标记）
      - line_labels: 对每行的 26 维多标签（用于 BCE Loss）
    """
    def __init__(self, ag_dataset, context_fraction=0.9, max_len=1024, path="gpt2", save_path=None, save_prompt=False, use_new_header=False):
        super().__init__()
        self.ag = ag_dataset
        self.context_fraction = context_fraction
        self.object_classes = self.ag.object_classes
        self.attn_rel_classes = ATTN_REL_CLASSES
        self.spat_rel_classes = SPAT_REL_CLASSES
        self.cont_rel_classes = CONT_REL_CLASSES
        self.relationship_classes = self.ag.relationship_classes
        self.samples = []
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.save_path = save_path
        self.save_prompt = save_prompt
        self.use_new_header = use_new_header
        self.check_relationship_ranges()
        self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回一个样本，字典包含：
          "prompt_text": str,
          "target_text": str,
          "line_texts": [str, str, ...],
          "line_labels": np.array(shape=[M, NUM_REL_CLASSES], dtype=float32)
        """
        return self.samples[idx]
    
    def check_relationship_ranges(self):
        max_attn = -1
        max_spat = -1
        max_cont = -1
        for vidx in range(len(self.ag)):
            gt_anno_video = self.ag.gt_annotations[vidx]
            for frame_data in gt_anno_video:
                objs = frame_data[1:]
                for obj in objs:
                    attn_ids = obj.get('attention_relationship', [])
                    spat_ids = obj.get('spatial_relationship', [])
                    cont_ids = obj.get('contacting_relationship', [])
                    if hasattr(attn_ids, 'tolist'):
                        attn_ids = attn_ids.tolist()
                    if hasattr(spat_ids, 'tolist'):
                        spat_ids = spat_ids.tolist()
                    if hasattr(cont_ids, 'tolist'):
                        cont_ids = cont_ids.tolist()
                    if attn_ids:
                        max_attn = max(max_attn, max(attn_ids))
                    if spat_ids:
                        max_spat = max(max_spat, max(spat_ids))
                    if cont_ids:
                        max_cont = max(max_cont, max(cont_ids))
        print(f"Max attention id: {max_attn}, Max spatial id: {max_spat}, Max contacting id: {max_cont}")

    def _extract_frame_number(self, frame_info):
        """从帧信息中提取帧号"""
        try:
            return int(frame_info.split('/')[-1].split('.')[0])
        except:
            return 0
        
    def _build_samples(self):
        for vidx in range(len(self.ag)):
            gt_anno_video = self.ag.gt_annotations[vidx]
            T = len(gt_anno_video)
            end = int(math.ceil(T * self.context_fraction)) - 1
            end = max(0, min(end, T - 1))
            if end >= T - 1:
                continue
            num_future = T - end - 1
            if num_future < 1:
                continue
            observed_anno = gt_anno_video[:end+1]
            obs_segments = self._merge_frames_for_objects(observed_anno)
            obs_by_obj = self._group_segments_by_object(obs_segments)
            if len(obs_segments) == 0:
                continue
            start_future = end + 1
            i = 0
            self.window_size = num_future
            while i < num_future:
                chunk_anno = gt_anno_video[start_future + i : start_future + i + self.window_size]
                if len(chunk_anno) == 0:
                    break
                fut_segments = self._get_individual_frame_segments(chunk_anno)
                fut_by_obj = self._group_segments_by_object(fut_segments)
                all_objects = set(obs_by_obj.keys()) | set(fut_by_obj.keys())
                future_frames = [self._extract_frame_number(frame_data[0]['frame']) for frame_data in chunk_anno]
                for obj_cls in all_objects:
                    obs_obj_segments = obs_by_obj.get(obj_cls, [])
                    fut_obj_segments = fut_by_obj.get(obj_cls, [])
                    if len(fut_obj_segments) == 0 or len(obs_obj_segments) == 0:
                        continue
                    prompt_text = self._build_text_from_segments_for_object(obs_obj_segments, obj_cls, observed=True)
                    target_text = self._build_text_from_segments_for_object(fut_obj_segments, obj_cls, observed=False, future_frames=future_frames)
                    line_texts, line_labels = self._make_line_texts_and_labels_for_object(fut_obj_segments, obj_cls)
                    out = self.truncate_prompt_only_if_needed(prompt_text, target_text)
                    if out is None:
                        continue
                    truncated_prompt, truncated_target = out

                    sample_dict = {
                        "video_index": vidx,
                        "object_class": obj_cls,
                        "prompt_text": truncated_prompt,
                        "target_text": target_text,
                        "line_texts": line_texts,
                        "line_labels": line_labels,
                        "future_frames": future_frames
                    }
                    self.samples.append(sample_dict)
                i += self.window_size
        print(f"[AGForLLM_ObjectCentric] total samples built: {len(self.samples)}")
    
    def truncate_prompt_only_if_needed(self, prompt_text, target_text):
        target_enc = self.tokenizer(target_text, add_special_tokens=False)
        if len(target_enc["input_ids"]) > self.max_len:
            return None
        combined_text = prompt_text + "\n" + target_text
        combined_enc = self.tokenizer(combined_text, add_special_tokens=False)
        if len(combined_enc["input_ids"]) <= self.max_len:
            return prompt_text, target_text
        prompt_lines = prompt_text.split('\n')
        truncated_prompt_lines = prompt_lines[:]
        while truncated_prompt_lines:
            candidate_prompt = "\n".join(truncated_prompt_lines)
            candidate_text = candidate_prompt + "\n" + target_text
            candidate_enc = self.tokenizer(candidate_text, add_special_tokens=False)
            if len(candidate_enc["input_ids"]) <= self.max_len:
                return candidate_prompt, target_text
            truncated_prompt_lines.pop(0)
        return None



    def _group_segments_by_object(self, segments):
        from collections import defaultdict
        obj_dict = defaultdict(list)
        for seg in segments:
            obj_cls = seg["object_class"]
            obj_dict[obj_cls].append(seg)
        for cls_ in obj_dict:
            obj_dict[cls_].sort(key=lambda x: x["start_time"])
        return dict(obj_dict)
    
    def _get_individual_frame_segments(self, chunk_anno):
        fut_segments = []
        for frame_data in chunk_anno:
            # 提取帧编号
            raw_frame_str = frame_data[0].get('frame', '')
            frame_num = self._extract_frame_number(raw_frame_str)
            # 获取该帧的对象信息
            objs = frame_data[1:]
            for obj_dict in objs:
                # 获取对象类别
                cls_idx = obj_dict.get('class', -1)
                if 0 <= cls_idx < len(self.object_classes):
                    obj_class = self.object_classes[cls_idx]
                else:
                    obj_class = "unknown"
                # 获取关系信息
                attn_ids = obj_dict.get('attention_relationship', [])
                spat_ids = obj_dict.get('spatial_relationship', [])
                cont_ids = obj_dict.get('contacting_relationship', [])
                # 将 numpy 数组转为列表（如果需要）
                if hasattr(attn_ids, 'tolist'):
                    attn_ids = attn_ids.tolist()
                if hasattr(spat_ids, 'tolist'):
                    spat_ids = spat_ids.tolist()
                if hasattr(cont_ids, 'tolist'):
                    cont_ids = cont_ids.tolist()
                # 为当前帧生成一个独立的段
                fut_segments.append({
                    "object_class": obj_class,
                    "attn_ids": attn_ids,
                    "spat_ids": spat_ids,
                    "cont_ids": cont_ids,
                    "start_time": frame_num,
                    "end_time": frame_num  # 每个帧独立，start_time 等于 end_time
                })
        return fut_segments
    
    def _merge_frames_for_objects(self, frames_annotation):
        segments = []
        running_dict = {}
        for f_idx, frame_data in enumerate(frames_annotation):
            raw_frame_str = frame_data[0].get('frame', '')
            filename = raw_frame_str.split('/')[-1]
            frame_num_str = filename.replace('.png', '')
            real_time = int(frame_num_str)
            objs = frame_data[1:]
            current_keys = set()
            for obj_dict in objs:
                cls_idx = obj_dict.get('class', -1)
                if 0 <= cls_idx < len(self.object_classes):
                    obj_class = self.object_classes[cls_idx]
                else:
                    obj_class = "unknown"
                attn_ids = obj_dict.get('attention_relationship', [])
                spat_ids = obj_dict.get('spatial_relationship', [])
                cont_ids = obj_dict.get('contacting_relationship', [])
                if hasattr(attn_ids, 'tolist'):
                    attn_ids = attn_ids.tolist()
                if hasattr(spat_ids, 'tolist'):
                    spat_ids = spat_ids.tolist()
                if hasattr(cont_ids, 'tolist'):
                    cont_ids = cont_ids.tolist()
                attn_tuple = tuple(sorted(attn_ids))
                spat_tuple = tuple(sorted(spat_ids))
                cont_tuple = tuple(sorted(cont_ids))
                key = (obj_class, attn_tuple, spat_tuple, cont_tuple)
                current_keys.add(key)
                if key not in running_dict:
                    running_dict[key] = {"start_time": real_time, "end_time": real_time}
                else:
                    running_dict[key]["end_time"] = real_time
            to_remove = []
            for key in running_dict:
                if key not in current_keys:
                    seg_info = running_dict[key]
                    segments.append({
                        "object_class": key[0],
                        "attn_ids": key[1],
                        "spat_ids": key[2],
                        "cont_ids": key[3],
                        "start_time": seg_info["start_time"],
                        "end_time": seg_info["end_time"],
                    })
                    to_remove.append(key)
            for rkey in to_remove:
                del running_dict[rkey]
        for key, seg_info in running_dict.items():
            segments.append({
                "object_class": key[0],
                "attn_ids": key[1],
                "spat_ids": key[2],
                "cont_ids": key[3],
                "start_time": seg_info["start_time"],
                "end_time": seg_info["end_time"],
            })
        segments.sort(key=lambda x: x["start_time"])
        return segments

    def _build_text_from_segments_for_object(self, obj_segments, obj_cls, observed=True, future_frames=None):
        if observed:
            # 对于观测段，保持基于合并区间的原有逻辑
            lines = []
            for i, seg in enumerate(obj_segments):
                if i > 0:
                    start_time = end_time
                else:
                    start_time = seg["start_time"]
                end_time = seg["end_time"] + 1
                line = self._construct_segment_text(start_time, end_time, seg, obj_cls, include_time=True, add_obj_marker=True, ignore_obj_mode=False)
                lines.append(line)
            return "\n".join(lines)
        else:
            # 对于未来段，根据 future_frames 为每个帧生成单独的文本
            if future_frames is None:
                raise ValueError("future_frames must be provided for future segments")
            lines = []
            for frame_num in future_frames:
                # 查找该帧对应的场景图段
                seg_for_frame = None
                for seg in obj_segments:
                    if seg["start_time"] <= frame_num <= seg["end_time"]:
                        seg_for_frame = seg
                        break
                if seg_for_frame is None:
                    # 未找到对应段时，使用默认文本
                    lines.append(f"Frame {frame_num}: No scene graph available.")
                else:
                    # 生成该帧的场景图文本
                    line = self._construct_segment_text(frame_num, frame_num, seg_for_frame, obj_cls, include_time=False, add_obj_marker=True, ignore_obj_mode=False)
                    lines.append(f"Frame {frame_num}: {line}")
            return "\n".join(lines)

    def _make_line_texts_and_labels_for_object(self, fut_obj_segments, obj_cls):
        line_texts = []
        line_labels = []
        for i, seg in enumerate(fut_obj_segments):
            if i > 0:
                start_time = end_time
            else:
                start_time = seg["start_time"]
            end_time = seg["end_time"] + 1
            line_str = self._construct_segment_text(start_time, end_time, seg, obj_cls, include_time=False, add_obj_marker=False, ignore_obj_mode=True)
            line_texts.append(line_str)
            row_label = np.zeros((len(self.relationship_classes),), dtype=np.float32)
            for rid in seg["attn_ids"]:
                if 0 <= rid < len(self.relationship_classes):
                    row_label[self.relationship_classes.index(self.attn_rel_classes[rid])] = 1
            for rid in seg["spat_ids"]:
                if 0 <= rid < len(self.relationship_classes):
                    row_label[self.relationship_classes.index(self.spat_rel_classes[rid])] = 1
            for rid in seg["cont_ids"]:
                if 0 <= rid < len(self.relationship_classes):
                    row_label[self.relationship_classes.index(self.cont_rel_classes[rid])] = 1
            line_labels.append(row_label)
        if len(line_labels) == 0:
            return [], np.zeros((0, len(self.relationship_classes)), dtype=np.float32)
        line_labels_arr = np.stack(line_labels, axis=0)
        return line_texts, line_labels_arr

    def _construct_segment_text(self, start_time, end_time, seg, obj_cls, include_time=False, add_obj_marker=True, ignore_obj_mode=False):
        """
        辅助函数：根据单个 segment 构造文本。
        
        参数：
         - seg: 字典，包含该段的信息（如 attn_ids、spat_ids、cont_ids、start_time、end_time）
         - obj_cls: 对象类别
         - include_time (bool): 是否在文本中包含时间信息，比如 "time [start..end]:"
         - add_obj_marker (bool): 是否在对象名称前添加特殊标记，例如 "<obj>"
        
        返回：
         - 构造好的字符串文本
        """
        # 将各关系 id 转为描述字符串
        attn_str = ",".join([self.attn_rel_classes[id_] for id_ in seg["attn_ids"]]) or "None"
        spat_str = ",".join([self.spat_rel_classes[id_] for id_ in seg["spat_ids"]]) or "None"
        cont_str = ",".join([self.cont_rel_classes[id_] for id_ in seg["cont_ids"]]) or "None"
        # 根据参数决定是否添加时间信息
        if start_time < end_time:
            time_text = f"Frame {start_time}..{end_time}: " if include_time else ""
        else:
            time_text = f"Frame {end_time}: " if include_time else ""

        # 根据参数决定对象文本的格式
        obj_text = f"object: {obj_cls}" if add_obj_marker else f"Object[{obj_cls}]"
        text = f"{time_text}{obj_text} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
        # text = f"{time_text}{obj_text} Person attention to the object: {attn_str}, the object located relative to person: {spat_str}, Person contact with the object: {cont_str}."
        # if not ignore_obj_mode:
        #     text = f"{time_text}{obj_text} Person attention to {obj_cls}: {attn_str}, {obj_cls} located relative to person: {spat_str}, Person contact with {obj_cls}: {cont_str}."
        # else:
        #     text = f"{time_text}Person attention to the object: {attn_str}, the object located relative to person: {spat_str}, Person contact with the object: {cont_str}."
        return text

#################################################
# 2. JointModel: base_model + classifier
#################################################
class JointModel(nn.Module):
    def __init__(self, base_model: nn.Module, classifier: nn.Module, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, attention_mask=None, labels=None, output_hidden_states=False, return_dict=False):
        lm_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return lm_outputs

#################################################
# 3. FineTuner: 初始化 -> 训练循环 与 损失计算
#################################################
class SceneGraphFineTuner:
    def __init__(self,
                 model_path,
                 local_rank,
                 world_size,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 learning_rate=5e-4,
                 lr_classify=5e-5,
                 epochs=3,
                 max_seq_length=2048,
                 alpha=1.0,
                 gamma=0.2,
                 decode_ratio=0.5,
                 decode_length=256,
                 gradient_accumulation_steps=1,
                 use_transition_loss=False,
                 transition_lambda=0.05,
                 tau=0.2,                 # 只对 abs(p_{t+1}-p_t) < tau 的关系做平滑
                 temp_lambda=1.0,         # KL 温度 (soften 分布)
                 ckpt_path=None,
                 enable_classifier=True,
                 object_classes=None,
                 phase="train",
                 save_prompt=False,
                 use_new_header=False,
                 save_path=None,
                 ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.learning_rate = learning_rate
        self.lr_classify = lr_classify
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.alpha = alpha
        self.gamma = gamma
        self.decode_ratio = decode_ratio
        self.decode_length = decode_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_classifier = enable_classifier
        self.ckpt_path = ckpt_path
        self.object_classes = object_classes
        self.phase = phase  # 设置模型的运行模式
        self.save_prompt = save_prompt
        self.save_path = save_path
        self.use_new_header = use_new_header
        self.model_device = torch.device("cuda", self.local_rank)
        self.tau = tau
        self.temp_lambda = temp_lambda

        # 1) 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 添加特殊标记
        
        special_tokens = {"additional_special_tokens": ["<obj>"]}
        self.tokenizer.add_special_tokens(special_tokens)

        # 2) 选择训练模式或推理模式
        if self.phase == "eval":
            self._initialize_for_evaluation(model_path)
        else:
            self._initialize_for_training(model_path, lora_r, lora_alpha, lora_dropout)

        self.use_transition_loss = use_transition_loss
        self.transition_lambda = transition_lambda

    def _initialize_for_evaluation(self, model_path):
        """初始化用于推理的模型，参考 SceneGraphAnticipator"""
        print("[Info] Initializing model in EVAL mode...")

        # 加载基础 CausalLM 模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(self.model_device)
        base_model.resize_token_embeddings(len(self.tokenizer))

        # 加载 LoRA
        peft_model = PeftModel.from_pretrained(base_model, self.ckpt_path).to(self.model_device)
        peft_model.eval()
        for p in peft_model.parameters():
            p.requires_grad_(False)

        # 初始化分类头
        hidden_size = peft_model.model.config.hidden_size
        classifier = nn.Linear(hidden_size, NUM_REL_CLASSES).to(self.model_device)

        # 加载分类头权重
        classifier_path = f"{self.ckpt_path}/classifier.bin"
        state_dict = torch.load(classifier_path, map_location=self.model_device)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)

        # 组合 JointModel
        self.joint_model = JointModel(peft_model, classifier, hidden_size).eval().to(self.model_device)

    def _initialize_for_training(self, model_path, lora_r, lora_alpha, lora_dropout):
        """初始化用于训练的模型"""
        print("[Info] Initializing model in TRAIN mode...")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            # device_map={"": f"cuda:{self.local_rank}"} if self.world_size > 1 else "auto"
        ).to(self.model_device)  # 先将模型整体移动到单个设备上
        base_model.resize_token_embeddings(len(self.tokenizer))

        if self.ckpt_path is None:
        # 配置 LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
                layers_to_transform=list(range(4, 28, 4))
            )
            peft_model = get_peft_model(base_model, lora_config)
            peft_model.train().to(self.model_device)

            # 初始化分类头
            hidden_size = peft_model.model.config.hidden_size
            classifier = nn.Linear(hidden_size, NUM_REL_CLASSES).to(self.model_device)
            self.joint_model = JointModel(peft_model, classifier, hidden_size).to(self.model_device)
        else:
            # 加载 LoRA 权重
            peft_model = PeftModel.from_pretrained(base_model, self.ckpt_path).to(self.model_device)
            peft_model.train().to(self.model_device)
            print("✓ LoRA 权重加载成功")
            
            for name, param in peft_model.named_parameters():
                if "lora" in name:  # 只对LoRA参数启用梯度
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            hidden_size = peft_model.model.config.hidden_size
            classifier = nn.Linear(hidden_size, NUM_REL_CLASSES).to(self.model_device)

            # 组合 JointModel
            self.joint_model = JointModel(peft_model, classifier, hidden_size).to(self.model_device)
            classifier_path = os.path.join(self.ckpt_path, "classifier.bin")
            if os.path.exists(classifier_path):
                self.joint_model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location=self.model_device)
                )
                print("✓ 分类器权重加载成功")
        self.classifier_loss_fn = nn.BCEWithLogitsLoss()

        # 调整 Tokenizer
        self.joint_model.base_model.resize_token_embeddings(len(self.tokenizer))
    
    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """加载模型检查点，包括LoRA权重和分类器权重（训练模式）"""
        print(f"正在从 {checkpoint_path} 加载检查点...")
        
        try:
            # 1. 加载分类器权重
            classifier_path = os.path.join(checkpoint_path, "classifier.bin")
            if os.path.exists(classifier_path):
                self.joint_model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location=self.model_device)
                )
                print("✓ 分类器权重加载成功")

            # 2. 加载LoRA适配器权重
            # 获取当前设备
            device_map = {"": self.model_device}
            
            # 保存原始模型
            original_model = self.joint_model.base_model
            
            # 使用PeftModel.from_pretrained加载LoRA权重
            self.joint_model.base_model = PeftModel.from_pretrained(
                original_model,
                checkpoint_path,
                device_map=device_map
            )
            
            # 确保模型处于训练状态
            self.joint_model.base_model.train()
            print("✓ LoRA适配器加载成功")

            # 3. 加载优化器和调度器状态(如果提供)
            if optimizer is not None and scheduler is not None:
                state_path = os.path.join(checkpoint_path, "training_state.pt")
                if os.path.exists(state_path):
                    state_dict = torch.load(state_path, map_location=self.model_device)
                    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                    scheduler.load_state_dict(state_dict['scheduler_state_dict'])
                    print("✓ 优化器和学习率调度器状态加载成功")
                    
        except Exception as e:
            print(f"加载检查点失败: {e}")
            import traceback
            traceback.print_exc()
    def build_prompt_for_scene_graph(self, observed_segments, object_class, relationship_categories, num_future_frames, future_frames, use_new_header = False):
        
        if use_new_header == False: 
            header = (
                "You are a scene graph anticipation assistant. Given a series of observed frames, your task is to predict the scene graphs for future frames.\n"
                "Each frame's scene graph consists of descriptions for each object present, including attention, spatial, and contact relationships with the person.\n"
                "The possible relationship categories are:\n"
                f"  Attention: {', '.join(relationship_categories.get('Attention', []))}\n"
                f"  Spatial: {', '.join(relationship_categories.get('Spatial', []))}\n"
                f"  Contact: {', '.join(relationship_categories.get('Contact', []))}\n"
            )
        else:
            header = (
                "You are a scene graph anticipation assistant. Your task is to predict how a person will interact with a given object in future frames, "
                "based on a set of observed frame segments.\n"
                "Each scene graph describes the relationship between the person and the target object, using three types of relations:\n"
                "  - Attention: whether the person is visually attending to the object\n"
                "  - Contact: whether the person is physically interacting with the object\n"
                "  - Spatial: the relative spatial position of the object with respect to the person\n"
                "The possible relationship categories are:\n"
                f"  Attention: {', '.join(relationship_categories.get('Attention', []))}\n"
                f"  Spatial: {', '.join(relationship_categories.get('Spatial', []))}\n"
                f"  Contact: {', '.join(relationship_categories.get('Contact', []))}\n"
                "Note: These annotations follow the Action Genome protocol, where 5 frames are uniformly sampled per action segment, "
                "and each frame is labeled with a scene graph capturing person-object interactions."
            )
        # instruction_0 = (
        #     f"\nPlease generate the future segment for object [{object_class}] "
        #     "in the same structured format as above. "
        #     "Do not add extra commentary; output exactly in the given style.\n"
        # )
        instruction_0 = (
        "Output one scene graph per frame in the following format:\n"
        "Frame <index>: object: <object class> Person attention to the object: <attention relationship>, the object located relative to person: <spatial relationship>, Person contact with the object: <contact relationship>\n"
        "Ensure each frame is on a separate line and no additional commentary is included.\n"
        )
        observed_text = f"Observed segment for object [{object_class}]:\n" + "\n".join(observed_segments) + "\n"
        # instruction = f"Future {num_future_frames} segments for object [{object_class}]:\n"
        future_frames_text = "Future frames " + ", ".join(map(str, future_frames)) + f" for object {object_class}:"+ "\n"
        # prompt = header + observed_text + instruction_0 + future_frames_text
        prompt = header + instruction_0 + observed_text + future_frames_text
        # prompt = header + instruction + observed_text + instruction
        return prompt
    
    def build_prompt_for_scene_grap_old(self, observed_segments, object_class, relationship_categories, num_future_frames):
        
        header = (
            "You are a scene graph anticipation assistant. In scene graph anticipation, you are given a series of observed frames containing a specific object. Your task is to predict how a person will interact with this object in the future.\n"
            "Note:\n"
            "Attention indicates whether the person is looking at the object.\n"
            "Contact indicates whether the person physically touches or interacts with the object.\n"
            "Spatial indicates the relative spatial position of the object with respect to the person.\n"
            "The possible relationship categories are:\n"
            f"  Attention: {', '.join(relationship_categories.get('Attention', []))}\n"
            f"  Spatial: {', '.join(relationship_categories.get('Spatial', []))}\n"
            f"  Contact: {', '.join(relationship_categories.get('Contact', []))}\n"
        )
        instruction_0 = (
            f"\nPlease generate the future segment for object [{object_class}] "
            "in the same structured format as above. "
            "Do not add extra commentary; output exactly in the given style.\n"
        )
        
        observed_text = f"Observed segment for object [{object_class}]:\n" + "\n".join(observed_segments) + "\n"
        instruction = f"Future {num_future_frames} segments for object [{object_class}]:\n"
        prompt = header + instruction_0 + observed_text + instruction
        return prompt

    def collate_fn(self, batch):
        input_ids_only_list = []
        input_ids_list = []
        label_ids_list = []
        all_obj_positions = []
        all_rel_labels = []
        max_m = 0

        relationship_categories = {
            "Attention": ATTN_REL_CLASSES,
            "Spatial": SPAT_REL_CLASSES,
            "Contact": CONT_REL_CLASSES
        }
        for sample in batch:
            observed_segments = sample["prompt_text"].split("\n")
            object_class = sample["object_class"]
            target_text = sample["target_text"]
            line_texts = sample["line_texts"]
            line_labels = sample["line_labels"]
            future_frames = sample["future_frames"]  # 如果数据集未提供 future_frames，则默认递增
            


            prompt_text = self.build_prompt_for_scene_graph(
                observed_segments,
                object_class,
                relationship_categories,
                num_future_frames=len(line_texts),
                future_frames = future_frames,
                use_new_header=self.use_new_header
            )

            prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False)
            prompt_ids = prompt_enc["input_ids"]
            prompt_len = len(prompt_ids)
            target_enc = self.tokenizer(target_text, add_special_tokens=False)
            target_ids = target_enc["input_ids"]
            full_ids = prompt_ids + target_ids
            max_len_val = self.max_seq_length
            # if len(full_ids) > max_len_val:
            #     full_ids = full_ids[:max_len_val]
            labels = [-100] * prompt_len + target_ids
            labels = labels[:len(full_ids)]  # 保证长度一致

            input_ids_list.append(full_ids)
            input_ids_only_list.append(prompt_ids)
            label_ids_list.append(labels)
            positions_for_sample = []
            for line_str in line_texts:
                line_enc = self.tokenizer(line_str, add_special_tokens=False)
                obj_id = self.tokenizer.convert_tokens_to_ids("<obj>")
                pos_in_line = -1
                for i_tok, tok_val in enumerate(line_enc["input_ids"]):
                    if tok_val == obj_id:
                        pos_in_line = i_tok
                        break
                positions_for_sample.append(pos_in_line)
            max_m = max(max_m, len(positions_for_sample))
            all_obj_positions.append(positions_for_sample)
            all_rel_labels.append(line_labels)
        if self.phase == 'train':
            padded_inp = self.tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt")
        else:
            padded_inp = self.tokenizer.pad({"input_ids": input_ids_only_list}, return_tensors="pt")
        padded_label = self.tokenizer.pad({"input_ids": label_ids_list}, return_tensors="pt")["input_ids"]
        input_ids_ = padded_inp["input_ids"]
        attn_mask = (input_ids_ != self.tokenizer.pad_token_id).long()
        batch_size = len(all_obj_positions)
        obj_positions_tensor = torch.full((batch_size, max_m), -1, dtype=torch.long)
        rel_label_tensor = torch.zeros((batch_size, max_m, NUM_REL_CLASSES), dtype=torch.float32)
        for b_idx in range(batch_size):
            line_pos = all_obj_positions[b_idx]
            line_labs = all_rel_labels[b_idx]
            M_b = len(line_pos)
            obj_positions_tensor[b_idx, :M_b] = torch.tensor(line_pos, dtype=torch.long)
            rel_label_tensor[b_idx, :M_b, :] = torch.from_numpy(line_labs)

        if self.save_prompt:
                save_path = os.path.join(self.save_path, f"prompt.json")
                self.save_json(observed_segments, prompt_text, target_text, save_path)

        return {
            "input_ids": input_ids_.to(self.model_device),
            "attention_mask": attn_mask.to(self.model_device),
            "labels": padded_label.to(self.model_device),
            "obj_positions": obj_positions_tensor.to(self.model_device),  # 保留字段（目前不再依赖 <obj>）
            "rel_label": rel_label_tensor.to(self.model_device),
            "line_texts": [sample["line_texts"] for sample in batch],  # 原始行文本列表，用于额外计算行级损失
            "prompt_texts": prompt_text,  # 原始 prompt 文本列表，用于日志记录
        }

    def save_json(self, observed_lines, prompt_text, target_text, save_path):
        """
        将prompt_text、observed_lines和target_text保存到JSON文件中
        如果文件存在，则追加数据；如果不存在，则创建新文件
        
        Args:
            observed_lines: 观察帧的描述列表
            prompt_text: 提示文本
            target_text: 目标文本
            save_path: 保存路径，应当包含文件名
        """
        import json
        import os
        
        data = {
            "observed_lines": observed_lines,
            "prompt_text": prompt_text,
            "target_text": target_text,
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 检查文件是否存在
        if os.path.exists(save_path):
            try:
                # 读取现有文件
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    
                # 判断是列表还是单个对象
                if isinstance(existing_data, list):
                    existing_data.append(data)
                else:
                    # 如果是单个对象，转换为列表
                    existing_data = [existing_data, data]
                    
                # 写入更新后的数据
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    
            except json.JSONDecodeError:
                # 文件存在但不是有效JSON，创建新文件
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump([data], f, ensure_ascii=False, indent=2)
        else:
            # 文件不存在，创建新文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump([data], f, ensure_ascii=False, indent=2)
                
        print(f"数据已保存到 {save_path}")

    def compute_weighted_ce_loss_debug(self, logits, labels, lent=10):
        # 对 logits 和 labels 进行移位：忽略第一个 token 的预测，使用后续 token 进行预测
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        batch_size = shift_logits.size(0)
        seq_len = shift_logits.size(1)
        total_loss = 0.0
        # 获取 "Frame" token 的 id（假设它只对应一个 token）
        frame_token_id = self.tokenizer.encode("Frame", add_special_tokens=False)[0]

        # 定义基于 cosine 的权重函数：t in [0,1] -> weight in [1, 0.5]
        def weight_func(t):
            # 当 t=0 时: cos(0)=1, w(0)=0.5+0.5*1=1
            # 当 t=1 时: cos(pi/2)=0, w(1)=0.5+0.5*0=0.5
            return 0.5 + 0.5 * math.cos(math.pi/2 * t)
        
        # 对每个样本单独计算
        for b in range(batch_size):
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            # 计算移位后每个 token 的 loss，形状：[seq_len]
            sample_loss = loss_fn(shift_logits[b].contiguous().view(-1, shift_logits.size(-1)),
                                    shift_labels[b].contiguous().view(-1))
            sample_loss = sample_loss.view(seq_len)
            # mask 有效 token
            mask = (shift_labels[b] != -100).float().to(self.model_device)
            
            # 获取当前样本中所有出现 "Frame" 的位置
            frame_positions = (shift_labels[b] == frame_token_id).nonzero(as_tuple=True)[0]
            if len(frame_positions) > lent:
                # 第 lent 个 Frame 的位置（索引从 0 开始）
                fifth_frame_pos = frame_positions[lent-1].item()
                # 初始化权重全1
                weights = torch.ones(seq_len, dtype=torch.float32, device=self.model_device)
                # 设定归一化区间：[fifth_frame_pos+1, target_end)
                target_end = int(mask.sum().item())
                denom = target_end - (fifth_frame_pos + 1)
                if denom <= 0:
                    denom = 1  # 避免除0
                for i in range(fifth_frame_pos + 1, target_end):
                    t_norm = (i - (fifth_frame_pos + 1)) / denom  # t_norm 在 [0,1]
                    weights[i] = weight_func(t_norm)
                # 使用 (loss * weights * mask) 求和，再除以 (weights * mask) 求和
                weighted_loss = (sample_loss * weights * mask.view(-1)).sum() / (weights * mask.view(-1)).sum()
                total_loss += weighted_loss
            else:
                # 如果没有足够的 Frame 信息，直接使用 unweighted loss
                weighted_loss = (sample_loss * mask.view(-1)).sum() / mask.sum()
                total_loss += weighted_loss
        return total_loss / batch_size

    def compute_losses(self, batch_data):
        """
        统一计算损失：
          1. 使用 teacher forcing 对输入 (input_ids, attention_mask, labels) 进行一次 forward，
             得到 LM loss。
          2. 对 batch_data["line_texts"] 进行额外 forward，对所有行文本编码，利用平均池化得到行级表示，
             经过分类头得到 logits，再计算：
               - 行级多标签 BCE 损失 (line_bce_loss)
               - 行级转移分布损失 (transition_loss)：统计每个 sample 内相邻行之间，每个关系的转移，
                 用 MSE 比较预测转移分布与真实转移分布。
          3. 返回 LM_loss, line_bce_loss, transition_loss, 以及总损失：
             total_loss = LM_loss + alpha * (line_bce_loss + transition_lambda * transition_loss)
             
        注意：此函数内部包含两次 forward：
           - 第一次 forward 用于 teacher forcing（LM loss）
           - 第二次 forward 用于对行文本编码（行级损失）
        """
        # ---------- 1. LM forward（teacher forcing）----------
        outputs = self.joint_model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"],
            labels=batch_data["labels"],
            output_hidden_states=True,
            return_dict=True
        )
        # logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        # labels = batch_data["labels"]  # [batch_size, seq_len]
        # lm_loss = self.compute_weighted_ce_loss_debug(logits, labels, lent=10)
        lm_loss = outputs.loss  # 自回归 (teacher forcing) loss

        # ---------- 2. 对行文本计算行级损失 ----------
        # 将所有样本的行文本合并
        all_line_texts = []
        sample_line_counts = []  # 记录每个 sample 的行数
        for sample_lines in batch_data["line_texts"]:
            sample_line_counts.append(len(sample_lines))
            all_line_texts.extend(sample_lines)
        if len(all_line_texts) == 0:
            line_bce_loss = torch.tensor(0.0, device=self.model_device)
            transition_loss = torch.tensor(0.0, device=self.model_device)
        else:
            # 对所有行文本编码
            encoded_lines = self.tokenizer(
                all_line_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.model_device)

            outputs_lines = self.joint_model(
                input_ids=encoded_lines["input_ids"],
                attention_mask=encoded_lines["attention_mask"],
                labels=None,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_lines = outputs_lines.hidden_states[-1]  # [total_lines, seq_len, H]
            attn_mask_lines = encoded_lines["attention_mask"].unsqueeze(-1).float()  # [total_lines, seq_len, 1]
            pooled_lines = (hidden_lines * attn_mask_lines).sum(dim=1) / (attn_mask_lines.sum(dim=1) + 1e-9)  # [total_lines, H]
            # 通过分类头获得 logits
            logits_all = self.joint_model.classifier(pooled_lines)  # [total_lines, NUM_REL_CLASSES]
            # 重新将 logits 与真实标签（batch_data["rel_label"]，形状 [B, max_m, NUM_REL_CLASSES]）对齐
            line_pointer = 0
            bce_loss_total = 0.0
            trans_loss_total = 0.0
            valid_samples = 0
            for count in sample_line_counts:
                if count < 1:
                    continue
                sample_logits = logits_all[line_pointer: line_pointer+count]  # [num_lines, NUM_REL_CLASSES]
                sample_labels = batch_data["rel_label"][valid_samples, :count, :]  # [num_lines, NUM_REL_CLASSES]
                # 行级 BCE loss
                bce_loss_sample = self.classifier_loss_fn(sample_logits, sample_labels)
                bce_loss_total += bce_loss_sample
                # 转移损失：仅在 sample 行数>=2时计算
                if count >= 2:
                    # >>> new-trans  ────────────────────────────────
                    sample_probs = torch.sigmoid(sample_logits).clamp(1e-5, 1-1e-5)  # [m,R]
                    trans_loss_sample = 0.0
                    valid_pairs = 0

                    for i_line in range(count-1):
                        p_t,  p_tp  = sample_probs[i_line], sample_probs[i_line+1]        # [R]
                        y_t,  y_tp  = sample_labels[i_line], sample_labels[i_line+1]      # hard 0/1

                        # 门控：只对变化小于 τ 的关系平滑
                        mask = (torch.abs(p_tp - p_t) < self.tau).float()                     # [R]
                        if mask.sum() < 1:
                            continue

                        # 构造 2-维分布 [¬r, r]
                        eps = 1e-5
                        # 构造分布并加 eps
                        gt_t  = torch.stack([1 - y_t ,  y_t ], dim=-1).clamp(eps, 1-eps)        # [R,2]
                        gt_tp = torch.stack([1 - y_tp,  y_tp], dim=-1).clamp(eps, 1-eps)

                        pr_t  = torch.stack([1 - p_t ,  p_t ], dim=-1).clamp(eps, 1-eps)
                        pr_tp = torch.stack([1 - p_tp, p_tp], dim=-1).clamp(eps, 1-eps)

                        # KL(pred || gt) : 先对预测取 log，再与 GT 做 KL
                        kl1 = F.kl_div(pr_t.log(),  gt_t,  reduction="batchmean")
                        kl2 = F.kl_div(pr_tp.log(), gt_tp, reduction="batchmean")
                        pair_loss = 0.5 * (kl1 + kl2) * mask.mean()

                        trans_loss_sample += pair_loss
                        valid_pairs += 1

                    if valid_pairs > 0:
                        trans_loss_total += trans_loss_sample / valid_pairs
                    # <<< new-trans

                line_pointer += count
                valid_samples += 1

            if valid_samples > 0:
                line_bce_loss  = bce_loss_total   / valid_samples
                transition_loss = trans_loss_total / valid_samples
            else:
                line_bce_loss  = torch.tensor(0.0, device=self.model_device)
                transition_loss = torch.tensor(0.0, device=self.model_device)

        # ---------- 3. 组合总损失 ----------
        if self.use_transition_loss:
            total_loss = lm_loss + self.alpha * (line_bce_loss + self.transition_lambda * transition_loss)
        else:
            total_loss = lm_loss + self.alpha * line_bce_loss
        # total_loss = lm_loss + self.alpha * (line_bce_loss + self.transition_lambda * transition_loss)
        return lm_loss, line_bce_loss, transition_loss, total_loss

    def train_loop(self, dataset, batch_size=2, warmup_steps=1000, save_path="./lora_finetuned"):
        from torch.nn.parallel import DistributedDataParallel as DDP
        self.ddp_model = DDP(
            self.joint_model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            # find_unused_parameters=True
        )
        self.ddp_model._set_static_graph()
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn
        )
        param_groups = [
            {'params': [p for n, p in self.ddp_model.module.base_model.named_parameters()],
             'lr': self.learning_rate},
            {'params': [p for n, p in self.ddp_model.module.classifier.named_parameters()],
             'lr': self.lr_classify}
        ]
        optimizer = AdamW(param_groups)
        total_steps = (len(loader) // self.gradient_accumulation_steps) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        global_step = 0
        for epoch in range(self.epochs):
            sampler.set_epoch(epoch)
            for step, batch_data in enumerate(tqdm(loader, desc=f"[DDP] Ep{epoch+1} rank={self.local_rank}")):
                # 调用统一计算损失的函数
                lm_loss, line_bce_loss, trans_loss, total_loss = self.compute_losses(batch_data)
                total_loss = total_loss / self.gradient_accumulation_steps
                total_loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % 50 == 0 and self.local_rank == 0:
                        print(f"[Rank0] ep={epoch+1}, step={global_step} | "
                              f"lm={lm_loss:.4f}, line_bce={line_bce_loss:.4f}, trans={trans_loss:.4f}, "
                              f"total={(total_loss.item()*self.gradient_accumulation_steps):.4f}")
            if self.local_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                # 总是保存为 checkpoint
                checkpoint_dir = os.path.join(save_path, "checkpoint")
                # self.save_checkpoint(checkpoint_dir, optimizer, scheduler)
                print(f"[Rank0] epoch {epoch+1} saved => {checkpoint_dir}")
                
                # 如果是 5 的倍数，额外保存一个带编号的里程碑检查点
                if (epoch + 1) % 1 == 0:
                    epoch_dir = os.path.join(save_path, f"epoch_{epoch+1}")
                    self.save_checkpoint(epoch_dir, optimizer, scheduler)
                    print(f"[Rank0] epoch {epoch+1} milestone saved => {epoch_dir}")
                    
    
    def save_checkpoint(self, checkpoint_path, optimizer, scheduler):
        os.makedirs(checkpoint_path, exist_ok=True)
        self.ddp_model.module.base_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        if self.ddp_model.module.classifier is not None:
            torch.save(self.ddp_model.module.classifier.state_dict(), os.path.join(checkpoint_path, "classifier.bin"))
        # 同时保存优化器和调度器状态
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(checkpoint_path, "training_state.pt"))
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def evaluate(self, eval_dataset):
        """
        Evaluate the model's performance on the given evaluation dataset.

        Args:
            eval_dataset (AGForLLM): The evaluation dataset instance.

        Prints:
            Accuracy metrics for attention, spatial, contact relationships, and overall accuracy.
        """
        # Perform evaluation only on rank 0 to avoid duplicate computations in DDP
        if self.local_rank != 0:
            return
        self.phase = 'eval'

        # Create DataLoader with batch_size=1
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        # Set model to evaluation mode
        self.joint_model.eval()

        # Initialize counters for accuracy computation
        sum_correct_attn = 0
        sum_num_pos_attn = 0
        sum_correct_spat = 0
        sum_num_pos_spat = 0
        sum_correct_cont = 0
        sum_num_pos_cont = 0

        # 定义每个关系类别的索引范围
        attn_slice = slice(0, 3)    # Attention: 0-2
        spat_slice = slice(3, 9)    # Spatial: 3-8
        cont_slice = slice(9, 26)   # Contact: 9-25


        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Extract data from batch (batch_size=1, so index 0)
                line_labels = batch['rel_label'].squeeze(0)  # Shape: [1, M, 26], where M is number of future frames
                M = line_labels.shape[0]
                prompt_text = batch['prompt_texts']

                # Step 2: Generate output and extract new text
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                max_attempts = 3  # 最大重试次数
                encoding_lines = []
                attempt = 0
                while not encoding_lines and attempt < max_attempts:
                    generated_ids = self.joint_model.base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=256,  # Adjust as needed
                        do_sample=True,     # Greedy decoding for consistency
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    generated_lines = generated_text.replace(prompt_text,"")
                    generated_lines = generated_lines.split('\n')  # Take up to M lines
                    # Process each generated line
                    for i in range(len(generated_lines)):
                        # Step 3: Parse generated line with regex
                        if i < len(generated_lines):
                            line_str = generated_lines[i].strip()
                            match = _PATTERN_LINE.search(line_str)
                            # if match:
                            #     _, attn_seq, spat_seq, cont_seq = match.groups()
                            #     new_line = (
                            #         "object: Person attention to object: " + attn_seq + "; "
                            #         "object located relative to person: " + spat_seq + "; "
                            #         "Person contact with object: " + cont_seq + "."
                            #     )
                            #     encoding_lines.append(new_line)
                            if match:
                                _, attn_seq, spat_seq, cont_seq = match.groups()
                                new_line = (
                                    # f"object: attention: {attn_seq}, spatial: {spat_seq}, contact: {cont_seq}."
                                    f"Person attention to the object: {attn_seq}, the object located relative to person: {spat_seq}, Person contact with the object: {cont_seq}."
                                )
                                encoding_lines.append(new_line)
                            else:
                                # Fallback if regex fails
                                new_line = line_str if line_str else ""
                        else:
                            new_line = encoding_lines[-1]  # Handle case where fewer lines are generated
                            encoding_lines.append(new_line)
                        if len(encoding_lines) == M:
                            break 
                    attempt += 1

                    # Step 4: Encode line and get logits
                 # 确保有处理后的行再进行编码
                if encoding_lines:
                    line_enc = self.tokenizer(encoding_lines, return_tensors="pt", padding=True).to(self.model_device)
                    outputs_line = self.joint_model.base_model(
                        input_ids=line_enc['input_ids'],
                        attention_mask=line_enc['attention_mask'],
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hidden_line = outputs_line.hidden_states[-1]  # [1, seq_len, H]
                    attn_mask_line = line_enc['attention_mask'].unsqueeze(-1).float()  # [1, seq_len, 1]
                    pooled_line = (hidden_line * attn_mask_line).sum(dim=1) / (attn_mask_line.sum(dim=1) + 1e-9)  # [1, H]
                    logits = self.joint_model.classifier(pooled_line)  # [1, 26]
                    
                    logits_attn = torch.softmax(logits[:, attn_slice], dim=-1)  # [M, 3]
                    logits_spat = torch.softmax(logits[:, spat_slice], dim=-1)  # [M, 6]
                    logits_cont = torch.softmax(logits[:, cont_slice], dim=-1)  # [M, 17]

                    # 处理每一行
                    for i in range(len(encoding_lines)):
                        # 确保不超过 ground truth 标签的行数
                        if i >= line_labels.shape[0]:
                            break
                        gt_labels = line_labels[i]  # 当前行的 ground truth 标签，形状 [26]

                        # 对每个关系类别分别计算准确率
                        for category, logits_cat, cat_slice in [
                            ('attn', logits_attn[i], attn_slice),
                            ('spat', logits_spat[i], spat_slice),
                            ('cont', logits_cont[i], cont_slice)
                        ]:
                            # 获取 ground truth 中正标签的索引
                            gt_pos_indices = torch.where(gt_labels[cat_slice] > 0.5)[0].tolist()  # 在类别范围内的索引
                            k = len(gt_pos_indices)  # 正标签数量

                            if k > 0:  # 只有当有正标签时才计算
                                # 从 logits 中获取 top-k 索引
                                _, top_k_indices = torch.topk(logits_cat, k)
                                top_k_indices = top_k_indices.tolist()

                                # 计算 top-k 索引与 ground truth 正标签索引的交集
                                correct = len(set(top_k_indices) & set(gt_pos_indices))

                                # 根据类别更新累加变量
                                if category == 'attn':
                                    sum_correct_attn += correct
                                    sum_num_pos_attn += k
                                elif category == 'spat':
                                    sum_correct_spat += correct
                                    sum_num_pos_spat += k
                                elif category == 'cont':
                                    sum_correct_cont += correct
                                    sum_num_pos_cont += k

        # Compute accuracies
        accuracy_attn = sum_correct_attn / sum_num_pos_attn if sum_num_pos_attn > 0 else 1.0
        accuracy_spat = sum_correct_spat / sum_num_pos_spat if sum_num_pos_spat > 0 else 1.0
        accuracy_cont = sum_correct_cont / sum_num_pos_cont if sum_num_pos_cont > 0 else 1.0
        sum_correct_total = sum_correct_attn + sum_correct_spat + sum_correct_cont
        sum_num_pos_total = sum_num_pos_attn + sum_num_pos_spat + sum_num_pos_cont
        accuracy_total = sum_correct_total / sum_num_pos_total if sum_num_pos_total > 0 else 1.0

        # Print results
        print(f"Evaluation Results:")
        print(f"Attention Accuracy: {accuracy_attn:.4f}")
        print(f"Spatial Accuracy: {accuracy_spat:.4f}")
        print(f"Contact Accuracy: {accuracy_cont:.4f}")
        print(f"Overall Accuracy: {accuracy_total:.4f}")
    
    def generate_text(self, prompts, max_new_tokens, temperature, top_p):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(self.model_device)
        with torch.no_grad():
            outputs = self.joint_model.base_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

#################################################
# 4. main
#################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='AG dataset root path')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--datasize', type=str, default='full')
    parser.add_argument('--script_require', action='store_true')
    parser.add_argument('--context_fraction', type=float, default=0.9)
    parser.add_argument('--llama_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./lora_finetuned_ddp')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1.0, help="balance factor for BCE classification loss vs LM loss")
    parser.add_argument('--gamma', type=float, default=0.2, help="penalty factor for decode repetition (or other mistakes)")
    parser.add_argument('--decode_ratio', type=float, default=0.5, help="ratio of samples to decode in each batch for penalty")
    parser.add_argument('--decode_length', type=int, default=128, help="max tokens to generate for penalty check")
    parser.add_argument('--max_input_length', type=int, default=1000, help="Maximum total tokens for prompt + target. If exceed, truncate from tail.")
    parser.add_argument('--lr_classify', type=float, default=5e-4, help="learning rate for classifier head")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="number of update steps to accumulate gradient")
    parser.add_argument('--use_transition_loss', action='store_true', help="use transition loss")
    parser.add_argument('--transition_lambda', type=float, default=0.05, help="transition loss weight")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--enable_classifier', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--save_prompt', action='store_true')
    parser.add_argument('--use_new_header', action='store_true')
    parser.add_argument('--tau', type=float, default=0.2, help='gating threshold for transition loss')
    parser.add_argument('--temp_lambda', type=float, default=1.0, help='temperature for KL in transition loss')
    args = parser.parse_args()

    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.local_rank = local_rank
        print(f"[DDP init] rank={rank}, local_rank={args.local_rank}, world_size={world_size}")
    else:
        world_size = 1
        rank = 0
        print("[Warning] WORLD_SIZE not found => single process")

    from dataloader.action_genome.ag_dataset import AG
    
    # Initialize AG dataset
    from dataloader.action_genome.ag_dataset import AG
    if args.evaluate:
        ag_dataset = AG(
            phase='test',
            datasize='mini',
            data_path=args.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
            script_require=args.script_require
        )
    else:
        ag_dataset = AG(
            phase=args.phase,
            datasize=args.datasize,
            data_path=args.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
            script_require=args.script_require
        )

    dataset_for_llm = AGForLLM(
        ag_dataset=ag_dataset,
        context_fraction=args.context_fraction,
        max_len=args.max_input_length,
        path=args.llama_path,
        save_path = args.save_path,
        save_prompt=args.save_prompt,
        use_new_header=args.use_new_header,
    )
    finetuner = SceneGraphFineTuner(
        model_path=args.llama_path,
        local_rank=args.local_rank,
        world_size=world_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        lr_classify=args.lr_classify,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        alpha=args.alpha,
        gamma=args.gamma,
        decode_ratio=args.decode_ratio,
        decode_length=args.decode_length,
        use_transition_loss=args.use_transition_loss,
        transition_lambda=args.transition_lambda,
        ckpt_path=args.ckpt_path,
        enable_classifier=args.enable_classifier,
        object_classes=ag_dataset.object_classes,
        save_prompt=args.save_prompt,
        use_new_header=args.use_new_header,
        save_path=args.save_path,
        tau=args.tau,
        temp_lambda=args.temp_lambda,
    )
    # Run training or evaluation
    if args.evaluate:
        finetuner.evaluate(dataset_for_llm)
    else:
        finetuner.train_loop(
            dataset_for_llm,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            save_path=args.save_path
        )
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        print("Done partial decode + penalty training.")

if __name__ == "__main__":
    main()