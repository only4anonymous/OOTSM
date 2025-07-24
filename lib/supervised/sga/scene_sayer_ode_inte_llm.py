#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from collections import defaultdict
from peft import PeftModel
# 假设下列模块在你的项目中可用
from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierTransformer, GetBoxes
from lib.word_vectors import obj_edge_vectors

# 预定义关系类别（示例）
REL_CLASSES = [
    'looking_at', 'not_looking_at', 'unsure',
    'above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in',
    'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on'
]
ATTN_REL_CLASSES = ['looking_at', 'not_looking_at', 'unsure']
SPAT_REL_CLASSES = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
CONT_REL_CLASSES = ['carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
                    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
                    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']
NUM_REL_CLASSES = len(REL_CLASSES)

def dict_to_cpu(d):
    """
    递归地将字典 d 中的所有 tensor 转移到 CPU 上。
    """
    if isinstance(d, torch.Tensor):
        return d.cpu()
    elif isinstance(d, dict):
        return {k: dict_to_cpu(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_to_cpu(x) for x in d]
    else:
        return d

############################################
# 1. SceneGraphAnticipator：LLM模块
############################################
class SceneGraphAnticipator:
    """
    负责利用预微调好的 LLM（例如 LLaMA + LoRA + classify head）生成未来帧关系的先验预测文本。
    所有与 prompt 构造、文本生成及解析均集中在此处。
    """
    def __init__(self, model_path, lora_path, classifier_path, device='cuda', FP16=False, obj_classes=None, rel_classes=None):
        self.device = device
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        # 添加特殊标记
        special_tokens = {"additional_special_tokens": ["<obj>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        # 加载 base CausalLM
        if not FP16:
            base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        base_model.resize_token_embeddings(len(self.tokenizer))
        # 加载 LoRA 模块（假设使用 peft）
        peft_model = PeftModel.from_pretrained(base_model, lora_path).to(device)
        for p in peft_model.parameters():
            p.requires_grad_(False)
        peft_model.eval()
        # 构建分类头，并加载权重
        hidden_size = peft_model.model.config.hidden_size
        classifier = nn.Linear(hidden_size, NUM_REL_CLASSES).to(device)
        state_dict = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)
        # 构建 JointModel
        self.joint_model = JointModel(peft_model, classifier, hidden_size).eval().to(device)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        self.attn_rel_classes = ATTN_REL_CLASSES
        self.spat_rel_classes = SPAT_REL_CLASSES
        self.cont_rel_classes = CONT_REL_CLASSES

        self.relationship_categories = {
            "Attention": ATTN_REL_CLASSES,
            "Spatial": SPAT_REL_CLASSES,
            "Contact": CONT_REL_CLASSES
            }
        

    def build_prompt_for_scene_graph(self, observed_segments, object_class, relationship_categories, num_future_frames=1):
        """
        构造用于 LLM 的 prompt。输入仅来自观测到的 scene graph（文本形式）。
        输出示例格式：
        "You are a scene graph anticipation assistant. In scene graph anticipation, you are given a series of observed frames (in text)
         containing a specific object. Your task is to predict the future relationships for that object.
         Relationship categories:
           Attention: looking_at, not_looking_at, unsure
           Spatial: above, beneath, in_front_of, behind, on_the_side_of, in
           Contact: carrying, covered_by, drinking_from, eating, have_it_on_the_back, holding, leaning_on, lying_on, not_contacting, other_relationship, sitting_on, standing_on, touching, twisting, wearing, wiping, writing_on
         Observed segment for object [cup]:
         <obj> cup Attention: looking_at, Spatial: behind, Contact: holding
         ...
         Future 2 segments for object [cup]:
        "
        """
        header = (
            "You are a scene graph anticipation assistant. In scene graph anticipation, you are given a series of observed frames (in text) "
            "containing a specific object. Your task is to predict the future relationships for that object.\n"
            "Relationship categories:\n"
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

    def generate_text(self, prompts, max_new_tokens=256, temperature=0.7, top_p=0.95):
        """
        输入 prompt(s)，调用 joint_model.base_model.generate 生成文本，返回生成文本（str 或 list[str]）。
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            single_input = True
        else:
            single_input = False
        enc = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.joint_model.base_model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded = [self.tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(outputs.size(0))]
        return decoded[0] if single_input else decoded
        

    def classify_text(self, lines):
        """
        对生成的每一行文本调用 joint_model 中的分类头，返回 shape [B, NUM_REL_CLASSES] 的 logits。
        """
        enc = self.tokenizer(lines, padding=True, truncation=True, return_tensors='pt').to(self.device)
        _, clf_logits = self.joint_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=None,
            output_hidden_states=True,
            return_dict=True,
            do_classification=True,
            classifier_positions="last_token"
        )
        return clf_logits


    def process_prior(self, observed_anno_list, windows, max_new_tokens=128):
        """
        给定 observed_anno_list（列表，每个元素是 observed_segments），批量构造 prompt，
        调用生成函数，再利用分类头将生成的文本转换为 logits 分布，
        返回形状为两层 dict 的 logits。
        """
        results = {}  # 存储最终结果

        # 遍历每一个 observed_anno
        for i_anno, observed_anno in enumerate(observed_anno_list):
            obs_segments = self._merge_frames_for_objects_inference(observed_anno)
            obs_by_obj   = self._group_segments_by_object_inference(obs_segments)
            # 获取最后一帧出现的 objects
            end_frame_objects = set()
            if len(observed_anno) > 0:
                last_frame = observed_anno[-1]
                for obj in last_frame[1:]:
                    if 'class' in obj:
                        cidx = obj['class']
                        if 0 <= cidx < len(self.obj_classes):
                            end_frame_objects.add(self.obj_classes[cidx])
            
            all_objects = end_frame_objects

            # ---------- (A) 合并 prompts ----------
            prompts = []
            obj_list = []
            for obj_cls in all_objects:
                obs_obj_segments = obs_by_obj.get(obj_cls, [])
                if len(obs_obj_segments) == 0:
                    continue

                # 构造 prompt
                observed_text = self._build_text_from_segments_for_object(obs_obj_segments, obj_cls, observed=True).split("\n")
                full_prompt = self.build_prompt_for_scene_graph(
                    observed_segments=observed_text,
                    object_class=obj_cls,
                    relationship_categories=self.relationship_categories,
                    num_future_frames=windows
                )
                prompts.append(full_prompt)
                obj_list.append(obj_cls)

            generated_texts_raw = self.generate_text(
                prompts=prompts,
                max_new_tokens=max_new_tokens,   # 或其他合适值
                temperature=0.7,
                top_p=0.95
            )
            
            # 这里肯定是 list[str]，因为 prompts 不止一个
            if isinstance(generated_texts_raw, str):
                generated_texts_raw = [generated_texts_raw]

            generated_texts = []
            for i, generated_text in enumerate(generated_texts_raw):
                generated_texts.append(generated_text.replace(prompts[i], ""))

            # 收集所有 lines 和对应的对象信息
            all_lines = []
            obj_indices = []  # 存储每个 line 对应的 obj_cls 的索引
            obj_cls_list = []  # 存储 obj_cls 的列表，用于索引
            # 逐对象解析
            for i_obj, gen_txt in enumerate(generated_texts):
                obj_cls = obj_list[i_obj]  # 获取当前对象类别
                gen_txt = gen_txt.replace(prompts[i_obj],"")
                lines_raw = self._split_generated_to_lines(gen_txt)
                # 提取可用行
                lines_parsed = self.extract_future_segments(lines_raw)

                # 添加重试机制
                retry_count = 0
                max_retries = 3  # 设置最大重试次数
                while not lines_parsed and retry_count < max_retries:
                    print(f"lines_parsed 为空，正在重试 ({retry_count + 1}/{max_retries})...")
                    # 重新生成文本
                    generated_text_raw = self.generate_text(
                        prompts=[prompts[i_obj]],  # 只重试当前对象的 prompt
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.95
                    )[0]  # 确保返回的是单个字符串

                    generated_text = generated_text_raw.replace(prompts[i_obj], "")
                    lines_raw = self._split_generated_to_lines(generated_text)
                    lines_parsed = self.extract_future_segments(lines_raw)
                    retry_count += 1

                # user要保证“截取需要预测帧数 n 行”，若 lines_parsed 太多，只取前 n 行
                # 如果不足 n 行，后面可以在 cat 时做对应
                needed = windows
                if len(lines_parsed) >= needed:
                    # 只取前 n 行
                    lines_use = lines_parsed[:needed]
                else:
                    lines_use = lines_parsed  # 可能不够
                
                all_lines.extend(lines_use)
                obj_indices.extend([i_obj] * len(lines_use))  # 记录每个 line 对应的 obj_cls 的索引
                obj_cls_list.append(obj_cls)
        
            # 对所有行文本 tokenize
            enc = self.tokenizer(
                all_lines,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            # 这里调用模型的 forward，获取最后一层隐藏状态
            outputs, _ = self.joint_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                output_hidden_states=True,
                return_dict=True,
                do_classification=False
            )

            # 获取 hidden states
            hidden_states = outputs.hidden_states[-1]  # 假设最后一层是 -1

            # 将 hidden states 分配回对应的对象
            line_idx = 0
            anno_results = {}
            for i_obj, gen_txt in enumerate(generated_texts):
                obj_cls = obj_list[i_obj]
                num_lines = obj_indices.count(i_obj)  # 获取当前对象对应的行数
                obj_hidden_states = hidden_states[line_idx:line_idx + num_lines]  # 提取 hidden states
                anno_results[obj_cls] = obj_hidden_states  # 存储结果
                line_idx += num_lines
            results[i_anno] = anno_results

        return results

        

    def _build_text_from_segments_for_object(self, obj_segments, obj_cls, observed=True):
        """
        原来用于生成整体 prompt 或 target 的文本（将多个段拼接成一段）
        这里调用 _construct_segment_text，不添加时间信息也不使用 <obj> 标记。
        """
        lines = []
        for i, seg in enumerate(obj_segments):
            if i>0:
                start_time = end_time
            else: 
                start_time = seg["start_time"]
            end_time = seg["end_time"]+1
            line = self._construct_segment_text(start_time, seg['end_time'], seg, obj_cls, include_time=False, add_obj_marker=True)
            lines.append(line)
        return "\n".join(lines)

    def _construct_segment_text(self, start_time, end_time, seg, obj_cls, include_time=False, add_obj_marker=False):
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
        attn_str = ",".join([self.rel_classes[id_] for id_ in seg["attn_ids"]]) or "None"
        spat_str = ",".join([self.rel_classes[id_] for id_ in seg["spat_ids"]]) or "None"
        cont_str = ",".join([self.rel_classes[id_] for id_ in seg["cont_ids"]]) or "None"
        # 根据参数决定是否添加时间信息
        if start_time < end_time:
            time_text = f"Frame {start_time}..{end_time}: " if include_time else ""
        else:
            time_text = f"Frame {end_time}: " if include_time else ""

        # 根据参数决定对象文本的格式
        obj_text = f"object: {obj_cls}" if add_obj_marker else f"Object[{obj_cls}]"
        return f"{time_text}{obj_text} Attention: {attn_str}, Spatial: {spat_str}, Contact: {cont_str}."

    def _split_generated_to_lines(self, generated_text):
        lines = generated_text.split("\n")
        return [ln.strip() for ln in lines if ln.strip()]

    def extract_future_segments(self, lines):
        pattern = re.compile(r"Attention:\s*(.*),\s*Spatial:\s*(.*),\s*Contact:\s*(.*)", flags=re.IGNORECASE)
        future_lines = []
        for line in lines:
            if pattern.search(line):
                future_lines.append(line)
        return future_lines
    
    def _merge_frames_for_objects_inference(self, frames_annotation):
        segments = []
        running_dict = {}
        for idx_frame, frame_data in enumerate(frames_annotation):
            raw_frame_str = frame_data[0].get('frame','')
            filename = raw_frame_str.split('/')[-1]
            frame_num_str = filename.replace('.png','')
            real_time = int(frame_num_str)
            objs = frame_data[1:]

            current_keys = set()
            for obj_dict in objs:
                cls_idx = obj_dict.get('class', -1)
                if 0 <= cls_idx < len(self.obj_classes):
                    obj_class = self.obj_classes[cls_idx]
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
                
                attn_abs = [self.rel_classes.index(self.attn_rel_classes[i]) for i in attn_ids]
                spat_abs = [self.rel_classes.index(self.spat_rel_classes[i]) for i in spat_ids]
                cont_abs = [self.rel_classes.index(self.cont_rel_classes[i]) for i in cont_ids]

                attn_tuple = tuple(sorted(attn_abs))
                spat_tuple = tuple(sorted(spat_abs))
                cont_tuple = tuple(sorted(cont_abs))
                

                key = (obj_class, attn_tuple, spat_tuple, cont_tuple)
                current_keys.add(key)

                if key not in running_dict:
                    running_dict[key] = {
                        "start_time": real_time,
                        "end_time": real_time
                    }
                else:
                    running_dict[key]["end_time"] = real_time

            to_remove = []
            for k_ in running_dict:
                if k_ not in current_keys:
                    seg_info = running_dict[k_]
                    segments.append({
                        "object_class": k_[0],
                        "attn_ids": list(k_[1]),
                        "spat_ids": list(k_[2]),
                        "cont_ids": list(k_[3]),
                        "start_time": seg_info["start_time"],
                        "end_time": seg_info["end_time"]
                    })
                    to_remove.append(k_)
            for kk in to_remove:
                del running_dict[kk]

        # flush
        for k_, seg_info in running_dict.items():
            segments.append({
                "object_class": k_[0],
                "attn_ids": list(k_[1]),
                "spat_ids": list(k_[2]),
                "cont_ids": list(k_[3]),
                "start_time": seg_info["start_time"],
                "end_time": seg_info["end_time"]
            })
        segments.sort(key=lambda x: x["start_time"])
        return segments

    def _group_segments_by_object_inference(self, segments):
        obj_dict = defaultdict(list)
        for seg in segments:
            obj_class = seg["object_class"]
            obj_dict[obj_class].append(seg)
        for oc in obj_dict:
            obj_dict[oc].sort(key=lambda x: x["start_time"])
        return dict(obj_dict)

############################################
# 2. JointModel：封装 base_model + classifier（用于 LLM部分）
############################################
class JointModel(nn.Module):
    def __init__(self, base_model: nn.Module, classifier: nn.Module, hidden_size: int):
        super(JointModel, self).__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.hidden_size = hidden_size

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                output_hidden_states=False,
                return_dict=False,
                do_classification=False,
                classifier_positions="last_token"):
        lm_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        clf_logits = None
        if do_classification:
            if not output_hidden_states:
                raise ValueError("If do_classification=True, must set output_hidden_states=True.")
            hidden_states = lm_outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            if classifier_positions == "last_token":
                pooled_emb = hidden_states[:, -1, :]
            else:
                pooled_emb = hidden_states.mean(dim=1)
            clf_logits = self.classifier(pooled_emb)
        return lm_outputs, clf_logits

class STTran(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None,
                 rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None, script_required=False, object_required=False, relation_required=False, llm_anticipator=None, window_size=5):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.script_required = script_required   # ScriptProcessor object
        self.object_required = object_required #对象表示处理单元（ORPU）：将对象特征与脚本嵌入融合，以生成增强的对象表示。
        self.relation_required = relation_required #关系预测模块：在预测对象之间的关系时，将脚本嵌入作为额外的输入特征，使关系预测能够结合脚本中的上下文信息。
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.num_features = 1936
        self.llm_anticipator = llm_anticipator  # 用于在 forward 内部调用

        self.object_classifier = ObjectClassifierTransformer(mode=self.mode, obj_classes=self.obj_classes)

        if self.script_required and self.object_required:
            self.get_subj_boxes = GetBoxes(1936+768)
            self.get_obj_boxes = GetBoxes(1936+768)
        elif self.script_required:
            self.get_subj_boxes = GetBoxes(1936+256)
            self.get_obj_boxes = GetBoxes(1936+256)
        else:
            self.get_subj_boxes = GetBoxes(1936)
            self.get_obj_boxes = GetBoxes(1936)
        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        self.subj_fc = nn.Linear(2376, 512)
        self.obj_fc = nn.Linear(2376, 512)
        self.vr_fc = nn.Linear(256 * 7 * 7, 512)
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        d_model = 1936
        if self.script_required and self.object_required:
            d_model += 768
        elif self.script_required:
            d_model += 256

        self.positional_encoder = PositionalEncoding(d_model, max_len=400)
        # temporal encoder
        global_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.global_transformer = Encoder(global_encoder, num_layers=3)
        # spatial encoder
        local_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.local_transformer = Encoder(local_encoder, num_layers=1)

        self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
        self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)

        # 如果需要脚本处理，添加 script_proj 和 relationship_head
        if self.script_required:
            script_embedding_dim = 768  # 根据使用的文本编码器（如 BERT）的输出维度调整
            self.script_proj = nn.Linear(script_embedding_dim, 256)  # 将脚本嵌入投影到256维
        
        self.cross_attn_layer = nn.MultiheadAttention(embed_dim=d_model,
                                                      num_heads=4,
                                                      batch_first=True)  # 仅示例
        self.window_size = window_size

    def get_scene_graph_labels(self, obj_indices, rel_indices):
        """
        获取Scene Graph中的节点标签和边的关系标签。
        
        Args:
            obj_indices: list[int]，Faster-RCNN生成的object标签索引。
            rel_indices: list[int]，预测生成的关系标签索引。
        
        Returns:
            dict: 包含object标签和relationship标签的字典。
        """
        object_labels = [self.obj_classes[obj_idx] for obj_idx in obj_indices]  # 获取对象标签
        relationship_labels = [self.rel_classes[rel_idx] for rel_idx in rel_indices]  # 获取关系标签

        scene_graph_info = {
            "objects": object_labels,
            "relationships": relationship_labels
        }
        
        return scene_graph_info
    
    def print_indices(self, entry):
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]  # 主体索引
        obj_class  = entry['pred_labels'][entry['pair_idx'][:, 1]]  # 客体索引
        # 这里仅举例: attention_distribution argmax当作“关系”
        attn_rel_indices = torch.argmax(entry["attention_distribution"], dim=1) 
        spaitial_rel_indices = torch.argmax(entry["spatial_distribution"], dim=1)
        contacting_rel_indices = torch.argmax(entry["contacting_distribution"], dim=1)

        return subj_class, obj_class, attn_rel_indices, spaitial_rel_indices, contacting_rel_indices

    def forward(self, entry, testing=False, window=5):
        entry = self.object_classifier(entry)
        # 判断是否需要处理脚本嵌入
        if self.script_required and "script_embeddings" in entry and entry["script_embeddings"] is not None:
            script_emb = entry["script_embeddings"]  # [768]
            script_emb = script_emb.unsqueeze(0)  # [1, 768]
            script_proj = self.script_proj(script_emb)  # [1, 256]
        else:
            script_proj = None  # 没有脚本嵌入时
        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        entry["subj_rep_actual"] = subj_rep
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        # 融合脚本嵌入与对象特征
        if self.script_required and script_proj is not None and self.object_required:
            num_objects = subj_rep.size(0)
            script_proj_relevant = script_proj.expand(num_objects, -1)  # [num_objects, 256]
            # 方法一：拼接脚本嵌入
            subj_rep = torch.cat([subj_rep, script_proj_relevant], dim=1)  # [num_objects, subj_feature_dim + 256]
            obj_rep = torch.cat([obj_rep, script_proj_relevant], dim=1)    # [num_objects, obj_feature_dim + 256]
            # 方法二（可选）：加权融合
            # subj_rep = subj_rep + script_proj_relevant * some_weight
            # obj_rep = obj_rep + script_proj_relevant * some_weight

            # 使用增强后的对象特征进行后续处理
            entry["subj_rep_actual"] = subj_rep
            entry["obj_rep_actual"] = obj_rep
            entry["script_proj"] = script_proj




        vr = self.union_func1(entry['union_feat']) + self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)

        # 融合脚本嵌入与关系特征
        if self.script_required and script_proj is not None and self.relation_required:
            num_relations = rel_features.size(0)
            script_proj_relevant = script_proj.expand(num_relations, -1)  # [num_relations, 256]
            # 方法一：拼接脚本嵌入
            rel_features = torch.cat([rel_features, script_proj_relevant], dim=1)  # [num_relations, feature_dim + 256]
            # 方法二（可选）：条件生成或加权融合
            # rel_features = rel_features + script_proj_relevant * some_weight
        
        # Spatial-Temporal Transformer
        # spatial message passing
        frames = []
        im_indices = entry["boxes"][
            entry["pair_idx"][:, 1], 0]  # im_indices -> centre cordinate of all objects in a video
        for l in im_indices.unique():
            frames.append(torch.where(im_indices == l)[0])
        frame_features = pad_sequence([rel_features[index] for index in frames], batch_first=True)
        masks = (1 - pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool()
        rel_ = self.local_transformer(frame_features, src_key_padding_mask=masks.cuda())
        rel_features = torch.cat([rel_[i, :len(index)] for i, index in enumerate(frames)])
        # subj_dec = self.subject_decoder(rel_features, src_key_padding_mask=masks.cuda())
        # subj_dec = subj_compress(subj_dec)
        # entry["subj_rep_decoded"] = subj_dec
        # entry["spatial_encoder_out"] = rel_features
        # temporal message passing
        sequences = []
        for l in obj_class.unique():
            k = torch.where(obj_class.view(-1) == l)[0]
            if len(k) > 0:
                sequences.append(k)
        pos_index = []
        for index in sequences:
            im_idx, counts = torch.unique(entry["pair_idx"][index][:, 0].view(-1), return_counts=True, sorted=True)
            counts = counts.tolist()
            pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
            pos_index.append(pos)

        sequence_features = pad_sequence([rel_features[index] for index in sequences], batch_first=True)
        in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]), diagonal=0)).type(
            torch.bool)
        # in_mask = (1-torch.ones(sequence_features.shape[1],sequence_features.shape[1])).type(torch.bool)
        in_mask = in_mask.cuda()
        masks = (1 - pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool()
        pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None
        sequence_features = self.positional_encoder(sequence_features, pos_index)
        # out = torch.zeros(sequence_features.shape)
        seq_len = sequence_features.shape[1]
        mask_input = sequence_features
        out = self.global_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)

        rel_ = out
        in_mask = None
        rel_ = rel_.cuda()
        rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(sequences, rel_)])
        rel_ = None
        indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, rel_features.shape[1])
        assert len(indices_flat) == len(entry["pair_idx"])
        global_output = torch.zeros_like(rel_features).to(rel_features.device)
        global_output.scatter_(0, indices_flat, rel_flat)
        entry["attention_distribution"] = self.a_rel_compress(global_output)
        entry["spatial_distribution"] = self.s_rel_compress(global_output)
        entry["contacting_distribution"] = self.c_rel_compress(global_output)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])
        # detached_outputs = global_output.clone().detach()
        entry["subject_boxes_dsg"] = self.get_subj_boxes(global_output)
        # entry["object_boxes_dsg"] = self.get_obj_boxes(global_output)

        pair_idx = entry["pair_idx"]
        boxes_rcnn = entry["boxes"]
        entry["global_output"] = global_output
        # entry["detached_outputs"] = detached_outputs
        entry["subject_boxes_rcnn"] = boxes_rcnn[pair_idx[:, 0], 1:].to(global_output.device)
        # entry["object_boxes_rcnn"] = boxes_rcnn[pair_idx[ :, 1], 1 : ].to(global_output.device)

        return entry
    


class get_derivatives(nn.Module):
    def __init__(self, script_required=False, object_required=False, relation_required=False):
        super(get_derivatives, self).__init__()
        
        dim = 1936
        middle_dim = 2048    
        if script_required and object_required:
            dim += 768
            middle_dim += 768
        elif script_required:
            dim += 256
            middle_dim += 256



        self.net = nn.Sequential(nn.Linear(dim, middle_dim), nn.Tanh(),
                                 nn.Linear(middle_dim, middle_dim), nn.Tanh(),
                                 nn.Linear(middle_dim, dim))

    def forward(self, t, y):

        out = self.net(y)
        return out



class SceneSayerODE(nn.Module):

    def __init__(self, mode, attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None,
                 rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None, max_window=None, script_required=False, object_required=False, relation_required=False, use_classify_head=False, llama_path=None,
                 lora_path=None,
                 classifier_path=None,
                 use_fusion=False,
                 save_path=None,
                 ):
        super(SceneSayerODE, self).__init__()
        self.mode = mode
        self.diff_func = get_derivatives(script_required, object_required, relation_required)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.d_model = 1936
        if script_required and object_required:
            self.d_model += 768
        elif script_required:
            self.d_model += 256
        self.max_window = max_window  
        self.dsgdetr = STTran(self.mode,
                              obj_classes=obj_classes,
                              rel_classes=rel_classes,
                              attention_class_num=attention_class_num,
                              spatial_class_num=spatial_class_num,
                              contact_class_num=contact_class_num,
                              script_required=script_required,
                              object_required=object_required,
                              relation_required=relation_required,
                              window_size=max_window)
        self.ctr = 0
        self.llm_anticipator = SceneGraphAnticipator(
            model_path=llama_path,
            lora_path=lora_path,
            classifier_path=classifier_path,
            device="cuda",
            FP16=False,
            obj_classes=obj_classes,
            rel_classes=rel_classes,
        )
        self.use_fusion = use_fusion    
        if self.use_fusion:
            # 定义一个线性层将 LLM embedding（3072维）映射到 d_model 维度
            self.llm_project = nn.Linear(3072, self.d_model, bias=False).to("cuda")
            # 如果需要更复杂的融合，可以加入一个融合层；这里示例采用拼接后过一层线性层
            self.fusion_layer = nn.Linear(2 * self.d_model, self.d_model, bias=True).to("cuda")
            self.save_path = save_path
            self.llm_results = {}
            # 检查 save_path 是否存在
            if self.save_path:
                import os
                # 构建文件名
                llm_results_file = os.path.join(self.save_path, "llm_results.pth")
                # 检查文件是否存在
                if os.path.exists(llm_results_file):
                    # 如果存在，则加载.将 llm_results 加载到 CPU 上
                    self.llm_results = torch.load(llm_results_file, map_location="cpu")
                    video_ids = " ".join(self.llm_results.keys())
                    print(f"Loaded LLM results from {llm_results_file} for video_ids: {video_ids}")
                else:
                    # 如果不存在，则创建目录
                    os.makedirs(self.save_path, exist_ok=True)
                    print(f"Created LLM results file at {llm_results_file}")



    def forward(self, entry, testing = False):
        entry = self.dsgdetr(entry)
        obj = entry["pair_idx"][ :, 1]
        if not testing:
            labels_obj = entry["labels"][obj]
        else:
            pred_labels_obj = entry["pred_labels"][obj]
            labels_obj = entry["labels"][obj]
        #pdb.set_trace()
        im_idx = entry["im_idx"]
        pair_idx = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds = im_idx.size(0)
        times = torch.tensor(entry["frame_idx"], dtype = torch.float32)
        indices = torch.reshape((im_idx[ : -1] != im_idx[1 : ]).nonzero(), (-1, )) + 1
        curr_id = 0
        times_unique = torch.unique(times).float()
        num_frames = len(gt_annotation)
        window = self.max_window
        if self.max_window == -1:
            window = num_frames - 1
        window = min(window, num_frames - 1)
        times_extend = torch.Tensor([times_unique[-1] + i + 1 for i in range(window)])
        global_output = entry["global_output"]
        anticipated_vals = torch.zeros(window, 0, self.d_model, device=global_output.device)
        #obj_bounding_boxes = torch.zeros(self.max_window, indices[-1], 4, device=global_output.device)
        frames_ranges = torch.cat((torch.tensor([0]).to(device=indices.device), indices, torch.tensor([num_preds]).to(device=indices.device)))
        frames_ranges = frames_ranges.long()
        k = frames_ranges.size(0) - 1
        for i in range(k - 1, 0, -1):
            diff = int(im_idx[frames_ranges[i]] - im_idx[frames_ranges[i - 1]])
            if diff > 1:
                frames_ranges = torch.cat((frames_ranges[ : i], torch.tensor([frames_ranges[i] for j in range(diff - 1)]).to(device=im_idx.device), frames_ranges[i : ]))
        if im_idx[0] > 0:
            frames_ranges = torch.cat((torch.tensor([0 for j in range(int(im_idx[0]))]).to(device=im_idx.device), frames_ranges))
        if frames_ranges.size(0) != num_frames + 1:
            frames_ranges = torch.cat((frames_ranges, torch.tensor([num_preds for j in range(num_frames + 1 - frames_ranges.size(0))]).to(device=im_idx.device)))
        entry["times"] = torch.repeat_interleave(times_unique.to(device=global_output.device), frames_ranges[1 : ] - frames_ranges[ : -1])
        entry["rng"] = frames_ranges
        times_unique = torch.cat((times_unique, times_extend)).to(device=global_output.device)

        # ========================================================
        # LLM fusion
        # ========================================================
        if self.use_fusion:
            video_id = entry.get("video_id", "unknown_video")  # 获取 video_id，如果不存在则使用默认值

            # 检查 self.llm_results 中是否存在当前 video_id 的结果
            if video_id in self.llm_results:
                # 如果存在，则直接使用
                object_hidden_dict = self.llm_results[video_id]
    
            else:
                # 如果不存在，则调用 LLM 进行推理
                observed_anno_list = []  # 存储所有 context 长度的 observed_anno

                for context_len in range(1, num_frames):  # 遍历每一帧作为 context 的最后一帧
                    # 截取当前 context 的数据
                    context_im_idx = im_idx[:frames_ranges[context_len]]
                    context_obj_class = labels_obj[:frames_ranges[context_len]]
                    # 注意力关系索引
                    context_attn_rel_indices = torch.argmax(entry["attention_distribution"][:frames_ranges[context_len]], dim=1)
                    # 空间关系索引
                    context_spatial_rel_indices = torch.argmax(entry["spatial_distribution"][:frames_ranges[context_len]], dim=1)
                    # 接触关系索引
                    context_contacting_rel_indices = torch.argmax(entry["contacting_distribution"][:frames_ranges[context_len]], dim=1)

                    observed_anno = self.build_frames_annotation(
                        context_im_idx,
                        context_obj_class,
                        context_attn_rel_indices,
                        context_spatial_rel_indices,
                        context_contacting_rel_indices
                    )
                    observed_anno_list.append(observed_anno)

                if self.llm_anticipator is not None:
                    object_hidden_dict = self.llm_anticipator.process_prior(
                        observed_anno_list=observed_anno_list,
                        windows=1  # 每次只预测未来一帧
                    )
                else:
                    object_hidden_dict = {}

                # 将结果记录在 self.llm_results 中
                self.llm_results[video_id] = dict_to_cpu(object_hidden_dict)

                # 保存 self.llm_results
                if self.save_path:
                    llm_results_file = os.path.join(self.save_path, "llm_results.pth")
                    torch.save(self.llm_results, llm_results_file)
                    # 输出 self.llm_results 的长度
                    print(f"Saved LLM results to {llm_results_file} for video_id: {video_id}, llm_results length: {len(self.llm_results)}")
        # --------------------- end ---------------------
        for i in range(1, window + 1):
            # masks for final output latents used during loss evaluation
            pred_indices_list = []
            mask_gt_list = []
            gt = gt_annotation.copy()
            for j in range(num_frames - i):
                if testing:
                    a, b = np.array(pred_labels_obj[frames_ranges[j] : frames_ranges[j + 1]].cpu()), np.array(labels_obj[frames_ranges[j + i] : frames_ranges[j + i + 1]].cpu())
                else:
                    a, b = np.array(labels_obj[frames_ranges[j] : frames_ranges[j + 1]].cpu()), np.array(labels_obj[frames_ranges[j + i] : frames_ranges[j + i + 1]].cpu())
                # persistent object labels
                intersection = np.intersect1d(a, b,  return_indices = False)
                ind1 = np.array([])        # indices of object labels from last context frame in the intersection
                ind2 = np.array([])        # indices of object labels that persist in the ith frame after the last context frame
                for element in intersection:
                    tmp1, tmp2 = np.where(a == element)[0], np.where(b == element)[0]
                    mn = min(tmp1.shape[0], tmp2.shape[0])
                    ind1 = np.concatenate((ind1, tmp1[ : mn]))
                    ind2 = np.concatenate((ind2, tmp2[ : mn]))
                L = []
                if testing:
                    ctr = 0
                    for detection in gt[i + j]:
                        if "class" not in detection.keys() or detection["class"] in intersection:
                            L.append(ctr)
                        ctr += 1
                    # stores modified ground truth
                    gt[i + j] = [gt[i + j][ind] for ind in L]
                ind1 = torch.tensor(ind1, dtype=torch.long, device=frames_ranges.device)
                ind2 = torch.tensor(ind2, dtype=torch.long, device=frames_ranges.device)
                # offset by subject-object pair position
                ind1 += frames_ranges[j]
                ind2 += frames_ranges[j + i]
                pred_indices_list.append(ind1)
                mask_gt_list.append(ind2)
            mask_preds = torch.cat(pred_indices_list, dim=0) if len(pred_indices_list) else torch.empty((0,), dtype=torch.long, device=frames_ranges.device)
            mask_gt = torch.cat(mask_gt_list, dim=0) if len(mask_gt_list) else torch.empty((0,), dtype=torch.long, device=frames_ranges.device)
            entry["mask_curr_%d" %i] = mask_preds
            entry["mask_gt_%d" %i] = mask_gt
            if testing:
                """pair_idx_test = pair_idx[mask_preds]
                _, inverse_indices = torch.unique(pair_idx_test, sorted=True, return_inverse=True)
                entry["im_idx_test_%d" %i] = im_idx[mask_preds]
                entry["pair_idx_test_%d" %i] = inverse_indices
                if self.mode == "predcls":
                    entry["scores_test_%d" %i] = entry["scores"][_.long()]
                    entry["labels_test_%d" %i] = entry["labels"][_.long()]
                else:
                    entry["pred_scores_test_%d" %i] = entry["pred_scores"][_.long()]
                    entry["pred_labels_test_%d" %i] = entry["pred_labels"][_.long()]
                if inverse_indices.size(0) != 0:
                    mx = torch.max(inverse_indices)
                else:
                    mx = -1
                boxes_test = torch.zeros(mx + 1, 5, device=entry["boxes"].device)
                boxes_test[torch.unique_consecutive(inverse_indices[: , 0])] = entry["boxes"][torch.unique_consecutive(pair_idx[mask_gt][: , 0])]
                boxes_test[inverse_indices[: , 1]] = entry["boxes"][pair_idx[mask_gt][: , 1]]
                entry["boxes_test_%d" %i] = boxes_test"""
                #entry["boxes_test_%d" %i] = entry["boxes"][_.long()]
                entry["last_%d" %i] = frames_ranges[-(i + 1)]
                mx = torch.max(pair_idx[ : frames_ranges[-(i + 1)]]) + 1
                entry["im_idx_test_%d" %i] = entry["im_idx"][ : frames_ranges[-(i + 1)]]
                entry["pair_idx_test_%d" %i] = entry["pair_idx"][ : frames_ranges[-(i + 1)]]
                if self.mode == "predcls":
                    entry["scores_test_%d" %i] = entry["scores"][ : mx]
                    entry["labels_test_%d" %i] = entry["labels"][ : mx]
                else:
                    entry["pred_scores_test_%d" %i] = entry["pred_scores"][ : mx]
                    entry["pred_labels_test_%d" %i] = entry["pred_labels"][ : mx]
                entry["boxes_test_%d" %i] = torch.ones(mx, 5).to(device=im_idx.device) / 2
                entry["gt_annotation_%d" %i] = gt
        #self.ctr += 1
        #anticipated_latent_loss = 0
        #targets = entry["detached_outputs"]
        for i in range(num_frames - 1):
            end = frames_ranges[i + 1]
            if curr_id == end:
                continue
            batch_y0 = global_output[curr_id : end]
            batch_obj = labels_obj[curr_id : end] # Get object labels for current frame
            batch_times = times_unique[i : i + window + 1]

            # LLM Fusion
            if self.use_fusion and i in object_hidden_dict:
                batch_y0 = self.fuse_global_with_llm(object_hidden_dict[i], batch_y0, batch_obj, i)

            ret = odeint(self.diff_func, batch_y0, batch_times, method='explicit_adams', options=dict(max_order=4, step_size=1))[1 : ]
            #ret = odeint(self.diff_func, batch_y0, batch_times, method='dopri5', rtol=1e-2, atol=1e-3)[1 : ]
            anticipated_vals = torch.cat((anticipated_vals, ret), dim=1)
            #obj_bounding_boxes[ :, curr_id : end, : ].data.copy_(self.dsgdetr.get_obj_boxes(ret))
            curr_id = end
        #for p in self.dsgdetr.get_subj_boxes.parameters():
        #    p.requires_grad_(False)
        entry["anticipated_subject_boxes"] = self.dsgdetr.get_subj_boxes(anticipated_vals)
        #for p in self.dsgdetr.get_subj_boxes.parameters():
        #    p.requires_grad_(True)
        entry["anticipated_vals"] = anticipated_vals
        entry["anticipated_attention_distribution"] = self.dsgdetr.a_rel_compress(anticipated_vals)
        entry["anticipated_spatial_distribution"] = torch.sigmoid(self.dsgdetr.s_rel_compress(anticipated_vals))
        entry["anticipated_contacting_distribution"] = torch.sigmoid(self.dsgdetr.c_rel_compress(anticipated_vals))
        
        #entry["anticipated_object_boxes"] = obj_bounding_boxes
        return entry

    def forward_single_entry(self, context_fraction, entry):
        # [0.3, 0.5, 0.7, 0.9]
        # end = 39
        # future_end = 140
        # future_frame_idx = [40, 41, .............139]
        # Take each entry and extrapolate it to the future
        # evaluation_recall.evaluate_scene_graph_forecasting(self, gt, pred, end, future_end, future_frame_idx, count=0)
        # entry["output"][0] = {pred_scores, pred_labels, attention_distribution, spatial_distribution, contact_distribution}

        assert context_fraction > 0
        entry = self.dsgdetr(entry)
        im_idx = entry["im_idx"]
        pair_idx = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds = im_idx.size(0)
        times = torch.tensor(entry["frame_idx"], dtype=torch.float32)
        indices = torch.reshape((im_idx[: -1] != im_idx[1:]).nonzero(), (-1,)) + 1
        times_unique = torch.unique(times).float()
        num_frames = len(gt_annotation)
        window = self.max_window
        if self.max_window == -1:
            window = num_frames - 1
        window = min(window, num_frames - 1)
        times_extend = torch.Tensor([times_unique[-1] + i + 1 for i in range(window)])
        global_output = entry["global_output"]
        frames_ranges = torch.cat(
            (torch.tensor([0]).to(device=indices.device), indices, torch.tensor([num_preds]).to(device=indices.device)))
        frames_ranges = frames_ranges.long()
        k = frames_ranges.size(0) - 1
        for i in range(k - 1, 0, -1):
            diff = int(im_idx[frames_ranges[i]] - im_idx[frames_ranges[i - 1]])
            if diff > 1:
                frames_ranges = torch.cat((frames_ranges[: i],
                                           torch.tensor([frames_ranges[i] for j in range(diff - 1)]).to(
                                               device=im_idx.device), frames_ranges[i:]))
        if int(im_idx[0]) > 0:
            frames_ranges = torch.cat(
                (torch.tensor([0 for j in range(int(im_idx[0]))]).to(device=im_idx.device), frames_ranges))
        if frames_ranges.size(0) != num_frames + 1:
            frames_ranges = torch.cat((frames_ranges, torch.tensor(
                [num_preds for j in range(num_frames + 1 - frames_ranges.size(0))]).to(device=im_idx.device)))
        entry["times"] = torch.repeat_interleave(times_unique.to(device=global_output.device),
                                                 frames_ranges[1:] - frames_ranges[: -1])
        entry["rng"] = frames_ranges
        num_frames = len(entry["gt_annotation"])
        pred = {}
        end = int(np.ceil(num_frames * context_fraction) - 1)
        while end > 0 and frames_ranges[end] == frames_ranges[end + 1]:
            end -= 1
        if end == num_frames - 1 or frames_ranges[end] == frames_ranges[end + 1]:
            return num_frames, pred
        
        # ========================================================
        # LLM fusion for forward_single_entry
        # ========================================================
        if self.use_fusion:
            video_id = entry.get("video_id", "unknown_video")  # 获取 video_id，如果不存在则使用默认值
            labels_obj = entry["labels"][entry["pair_idx"][:, 1]]  # 获取对象标签

            # # 检查 self.llm_results 中是否存在当前 video_id 的结果
            # if video_id in self.llm_results:
            #     # 如果存在，则直接使用
            #     object_hidden_dict = self.llm_results[video_id]
            #     print(f"Using cached LLM results for video_id: {video_id} in forward_single_entry")
            # else:
                # 如果不存在，则调用 LLM 进行推理
            observed_anno_list = []  # 存储所有 context 长度的 observed_anno

            # 截取当前 context 的数据
            context_im_idx = im_idx[:frames_ranges[end]]
            context_obj_class = labels_obj[:frames_ranges[end]]
            # 注意力关系索引
            context_attn_rel_indices = torch.argmax(entry["attention_distribution"][:frames_ranges[end]], dim=1)
            # 空间关系索引
            context_spatial_rel_indices = torch.argmax(entry["spatial_distribution"][:frames_ranges[end]], dim=1)
            # 接触关系索引
            context_contacting_rel_indices = torch.argmax(entry["contacting_distribution"][:frames_ranges[end]], dim=1)

            observed_anno = self.build_frames_annotation(
                context_im_idx,
                context_obj_class,
                context_attn_rel_indices,
                context_spatial_rel_indices,
                context_contacting_rel_indices
            )
            observed_anno_list.append(observed_anno)

            if self.llm_anticipator is not None:
                object_hidden_dict = self.llm_anticipator.process_prior(
                    observed_anno_list=observed_anno_list,
                    windows=1  # 每次只预测未来一帧
                )
            else:
                object_hidden_dict = {}

            # 将结果记录在 self.llm_results 中
            self.llm_results[video_id] = object_hidden_dict

                # # 保存 self.llm_results
                # if self.save_path:
                #     llm_results_file = os.path.join(self.save_path, "llm_results.pth")
                #     torch.save(self.llm_results, llm_results_file)
                #     print(f"Saved LLM results to {llm_results_file} for video_id: {video_id} in forward_single_entry")
        # --------------------- end ---------------------

        # ODE 积分之前进行 LLM 融合
        if self.use_fusion:
            batch_y0 = entry["global_output"][frames_ranges[end] : frames_ranges[end + 1]]
            batch_obj = entry["labels"][entry["pair_idx"][:, 1]][frames_ranges[end] : frames_ranges[end + 1]]  # Get object labels for current frame
            if object_hidden_dict and 0 in object_hidden_dict:  # 假设 object_hidden_dict 的 key 是帧索引
                batch_y0 = self.fuse_global_with_llm(object_hidden_dict[0], batch_y0, batch_obj, 0)

        ret = odeint(self.diff_func, batch_y0, times[end : ], method='explicit_adams', options=dict(max_order=4, step_size=1))[1 : ]
        # ret = odeint(self.diff_func, entry["global_output"][frames_ranges[end] : frames_ranges[end + 1]], times[end : ], method='explicit_adams', options=dict(max_order=4, step_size=1))[1 : ]
        pred["attention_distribution"] = torch.flatten(self.dsgdetr.a_rel_compress(ret), start_dim=0, end_dim=1)
        pred["spatial_distribution"] = torch.flatten(torch.sigmoid(self.dsgdetr.s_rel_compress(ret)), start_dim=0, end_dim=1)
        pred["contacting_distribution"] = torch.flatten(torch.sigmoid(self.dsgdetr.c_rel_compress(ret)), start_dim=0, end_dim=1)
        if self.mode == "predcls":
            pred["scores"] = entry["scores"][torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) : torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1].repeat(num_frames - end - 1)
            pred["labels"] = entry["labels"][torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) : torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1].repeat(num_frames - end - 1)
        else:
            pred["pred_scores"] = entry["pred_scores"][torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) : torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1].repeat(num_frames - end - 1)
            pred["pred_labels"] = entry["pred_labels"][torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) : torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1].repeat(num_frames - end - 1)
        pred["im_idx"]  = torch.tensor([i for i in range(num_frames - end - 1) for j in range(frames_ranges[end + 1] - frames_ranges[end])], dtype=torch.int32).to(device=frames_ranges.device)
        mx = torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) - torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1
        pred["pair_idx"] = (pair_idx[frames_ranges[end] : frames_ranges[end + 1]] - torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]])).repeat(num_frames - end - 1, 1) + mx * torch.reshape(pred["im_idx"], (-1, 1))
        pred["boxes"] = torch.ones(mx * (num_frames - end - 1), 5).to(device=im_idx.device) / 2
        return end + 1, pred

    def fuse_global_with_llm(self, object_hidden_dict, global_output, pred_labels, frame_index):
        """
        输入：
        - object_hidden_dict: dict, key 为 observed_anno 的序号, value 为该序号对应的 LLM 输出，
          其中 LLM 输出是一个字典，key 为 object name, value 为该对象的 embedding
        - global_output: tensor, shape [N, d_model]，其中 N 表示所有观测到的对象
        - pred_labels: tensor, shape [N]，每个元素为对象在 self.obj_classes 中的索引
        - frame_index: 当前帧的索引，用于从 object_hidden_dict 中获取对应的 LLM 输出

        处理流程：
        1. 对于当前帧的每个对象，从 object_hidden_dict 中获取对应的 LLM 输出。
        2. 将 global_output 中该对象的状态向量与 LLM 预测融合。

        返回融合后的 global_output（相同 shape）
        """
        fused_global = global_output.clone()
        # 遍历当前帧的每个对象
        for i, obj_id in enumerate(pred_labels):
            # 将 object id 映射为 object name
            obj_name = self.obj_classes[obj_id]
            # 检查 object_hidden_dict 中是否存在该对象的预测
            if obj_name not in object_hidden_dict:
                continue

            hidden_tensor = object_hidden_dict[obj_name]
            # hidden_tensor 的 shape 假设为 [1, O, 3072]
            T, O, D_llm = hidden_tensor.shape
            # 先在 O 维度取平均：得到每个时间步的平均向量， shape => [T, 3072]
            avg_over_obj = hidden_tensor.mean(dim=1)
            # 将 weighted_avg 通过 llm_project 映射到 d_model 维度
            mapped_llm = self.llm_project(avg_over_obj.to(global_output.device))  # [1, d_model]
            # 取 global_output 中对应的向量
            y_vec = global_output[i].unsqueeze(0)  # [1, d_model]
            # 融合方式：拼接后过 fusion_layer
            fused_vec = self.fusion_layer(torch.cat([y_vec, mapped_llm], dim=1))  # [1, d_model]
            # 更新 fused_global 对应位置
            fused_global[i] = fused_vec.squeeze(0)
        return fused_global
    
    def build_frames_annotation(
        self,
        im_idx: torch.Tensor,
        obj_class: torch.Tensor,
        attn_rel_indices: torch.Tensor,
        spatial_rel_indices: torch.Tensor,
        contacting_rel_indices: torch.Tensor
    ) -> list:
        """
        将输入转化为 frames_annotation 格式。

        输入：
            im_idx: [N], tensor, 每个元素代表一个对象所属的帧索引。
            obj_class: [N], tensor, 每个元素代表对象的类别索引。
            attn_rel_indices: [N], tensor, 每个元素是 list 或 tensor, 注意力关系索引。
            spatial_rel_indices: [N], tensor, 每个元素是 list 或 tensor, 空间关系索引。
            contacting_rel_indices: [N], tensor, 每个元素是 list 或 tensor, 接触关系索引。

        返回：
            frames_annotation: list, 每个元素对应一个帧的数据。
                每个帧数据是一个列表：
                    - 第一个元素是 {'frame': 'path/to/frame{im_idx[i]}.png'}
                    - 后续元素是对象的注释字典：
                        {
                            'class': <int>,
                            'attention_relationship': [...],
                            'spatial_relationship': [...],
                            'contacting_relationship': [...]
                        }
        """
        # 确保所有输入都是一维张量
        assert im_idx.dim() == 1, "im_idx 必须是一维张量"
        assert obj_class.dim() == 1, "obj_class 必须是一维张量"
        assert attn_rel_indices.dim() == 1, "attn_rel_indices 必须是一维张量"
        assert spatial_rel_indices.dim() == 1, "spatial_rel_indices 必须是一维张量"
        assert contacting_rel_indices.dim() == 1, "contacting_rel_indices 必须是一维张量"

        # 转换张量为列表
        im_idx = im_idx.tolist()
        obj_class = obj_class.tolist()
        attn_rel_indices = attn_rel_indices.tolist()
        spatial_rel_indices = spatial_rel_indices.tolist()
        contacting_rel_indices = contacting_rel_indices.tolist()

        # 按帧索引分组对象
        frames_dict = defaultdict(list)
        for i in range(len(im_idx)):
            frame_id = im_idx[i]
            obj_dict = {
                'class': obj_class[i],
                'attention_relationship': [attn_rel_indices[i]],
                'spatial_relationship': [spatial_rel_indices[i]],
                'contacting_relationship': [contacting_rel_indices[i]]
            }
            frames_dict[frame_id].append(obj_dict)

        # 按帧顺序组装 frames_annotation
        frames_annotation = []
        sorted_frame_ids = sorted(frames_dict.keys())
        for frame_id in sorted_frame_ids:
            frame_meta = {'frame': f'path/to/{frame_id}.png'}
            frame_objs = frames_dict[frame_id]
            frame_entry = [frame_meta] + frame_objs
            frames_annotation.append(frame_entry)

        return frames_annotation