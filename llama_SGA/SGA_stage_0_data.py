#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
import random
from collections import defaultdict

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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import sys

# 项目根目录
project_root = "your/project/path"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataloader.action_genome.ag_dataset import AG
from transformers import StoppingCriteria, StoppingCriteriaList

# 关系类别常量
REL_CLASSES = [
    'looking_at', 'not_looking_at', 'unsure',
    'above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in',
    'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on'
]

ATTN_REL_CLASSES = ['looking_at', 'not_looking_at', 'unsure']
SPAT_REL_CLASSES = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
CONT_REL_CLASSES = [
    'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on'
]
NUM_REL_CLASSES = len(REL_CLASSES)

# 用于解析生成文本的正则表达式
_PATTERN_LINE = re.compile(
    r'object:\s*([^P]+?)(?=\s*Person)\s*Person attention to [^:]+:\s*([^,]+),\s*[^:]+?\s*located relative to person:\s*([^,]+),\s*Person contact with [^:]+:\s*([^,\.]+)',
    re.IGNORECASE
)

def time_encoding(frame_id, fps=24):
    # 将帧ID转换为秒数
    time_in_seconds = frame_id / fps
    # 创建多尺度时间表示
    return {
        "absolute_time": time_in_seconds,
        "minute_mark": int(time_in_seconds // 60),
        "second_mark": time_in_seconds % 60
    }

# ===================  放在文件开头 (import 之后)  ===================
TAIL_FREQ_THRESHOLD = 300   # ≤ 此出现次数的类别被视为 tail
TAIL_DUP_FACTOR     = 3     # 含 tail 类样本额外复制次数
AUG_DROP_PROB       = 0.5   # 执行随机删帧数据增强的概率
AUG_MAX_DROP_FRAMES = 3     # 最多随机删掉多少帧
# ===================================================================

def set_global_seed(seed: int = 42, cuda_deterministic: bool = True) -> None:
    """
    设置 Python / NumPy / PyTorch / CuDNN 的随机种子。

    Args:
        seed (int): 全局种子数。
        cuda_deterministic (bool): True 时禁用 CuDNN 的非确定性算法，
                                   可完全复现；False 时保留 CuDNN 自动
                                   优化（速度更快，但可能有微小差异）。
    """
    import os, random, numpy as np, torch

    # 1) Python 内置
    random.seed(seed)

    # 2) Numpy
    np.random.seed(seed)

    # 3) PyTorch CPU / CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 所有 GPU

    # 4) CuDNN
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True 会让 CuDNN 根据输入自动寻找最快算法
        # 若要绝对可复现，请保持 False
        torch.backends.cudnn.benchmark = True

    # 5) 环境变量（部分库会读取）
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed] Global seed set to {seed}")

# set_global_seed(42)
#################################################
# 1. 自定义数据集类：AGForLLM_ObjectPrediction
#################################################
class AGForLLM_ObjectPrediction(Dataset):
    """
    Dataset class for the future frame object prediction task.
    Inputs:
      - Observed scene information from past frames (all objects' scene graphs merged by identical frames).
      - A list of future frame numbers.
    Outputs:
      - prompt_text: A text that includes the observed scene information and the future frame numbers.
      - target_text: For each future frame, the ground truth object list in the format:
                     "Frame X: object1, object2, ..." (each frame on a separate line).
    """
    def __init__(self, ag_dataset, context_fraction=0.9, max_len=1024, path="gpt2", video_id = False, include_timestamps=False, subject_id = False):
        super().__init__()
        self.ag = ag_dataset
        self.context_fraction = context_fraction
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.object_classes = self.ag.object_classes
        self.samples = []
        self.attn_rel_classes = ATTN_REL_CLASSES
        self.spat_rel_classes = SPAT_REL_CLASSES
        self.cont_rel_classes = CONT_REL_CLASSES
        self.relationship_classes = self.ag.relationship_classes
        self.include_timestamps = include_timestamps
        self.subject_id = subject_id
        # self._build_class_frequency()        # ### MOD START
        self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _build_samples(self):
        fps = 24  # 假设24帧每秒
        for vidx in range(len(self.ag)):
            gt_anno_video = self.ag.gt_annotations[vidx]
            T = len(gt_anno_video)
            # 使用前 context_fraction 的帧作为观测帧，其余作为未来帧
            end = int(math.ceil(T * self.context_fraction)) - 1
            end = max(0, min(end, T - 1))
            if end >= T - 1:
                continue
            observed_anno = gt_anno_video[:end + 1]
            length = 10
            # future_anno = gt_anno_video[end + 1:min(end + 1 + length, len(gt_anno_video))]
            future_anno = gt_anno_video[end + 1:]
            if len(future_anno) < 1:
                continue

            ### MOD START: 随机删除一段观测帧（数据增强）
            # if (random.random() < AUG_DROP_PROB and
            #     len(observed_anno) > AUG_MAX_DROP_FRAMES + 1):
            #     drop_n = random.randint(1, AUG_MAX_DROP_FRAMES)
            #     # 这里选择“剪掉末尾帧”策略；如需随机段或开头段，可自行替换
            #     observed_anno = observed_anno[:-drop_n]
            #     # 同步修正 end 索引
            #     end -= drop_n
            #     if end < 0:
            #         continue
            ### MOD END

            # 获取未来帧编号
            # if self.video_id:
            #     video_id = [self._extract_frame_number(frame_data[0]['frame'], video_id = self.video_id) for frame_data in future_anno]
            #     future_frames = [self._extract_frame_number(frame_data[0]['frame']) for frame_data in future_anno]
            # else:
            future_frames = [self._extract_frame_number(frame_data[0]['frame']) for frame_data in future_anno]
            video_id = [self._extract_frame_number(frame_data[0]['frame'], video_id = True) for frame_data in future_anno][0]
            if self.ag.script_require:
                script = self.ag.get_script(video_id)
                if not script:
                    continue
            if self.subject_id:
                subject, scene = self.ag.get_subject_scene(video_id)

            if len(future_frames) < 1:
                continue

            # 构建 observed_text：直接聚合所有观测帧的scene graph并合并相同帧
            observed_text = self._aggregate_scene_graph(observed_anno, include_timestamps=self.include_timestamps)
            if not observed_text.strip():
                continue

            # 构建 prompt_text
            prompt_text = self._build_prompt(observed_text, future_frames, observed_anno, max_length=self.max_len, script=self.ag.script_require, include_timestamps=self.include_timestamps, subject_id = self.subject_id)

            # 构建 target_text：针对每个未来帧，列出该帧中的所有物体
            target_lines = []
            if self.ag.script_require:
                target_lines.append(f"Script: {script}")
                target_lines.append(f"Objects prediction: ")
            if self.subject_id:
                target_lines.append(f"The subject index is {subject}")
                target_lines.append(f"The scene is {scene}")
            for frame_data in future_anno:
                frame_info = frame_data[0]
                frame_num = self._extract_frame_number(frame_info['frame'])
                objs_in_frame = []
                for obj_dict in frame_data[1:]:
                    cls_idx = obj_dict.get('class', -1)
                    if 0 <= cls_idx < len(self.object_classes):
                        cur_obj = self.object_classes[cls_idx]
                    else:
                        cur_obj = "unknown"
                    if cur_obj not in objs_in_frame:
                        objs_in_frame.append(cur_obj)
                if not objs_in_frame:
                    continue
                
                if self.include_timestamps:
                    time_in_seconds = frame_num / fps
                    target_lines.append(f"Frame {frame_num} [T={time_in_seconds:.2f}s]: {', '.join(objs_in_frame)}")
                else:
                    target_lines.append(f"Frame {frame_num}: {', '.join(objs_in_frame)}")
            if not target_lines:  # 如果目标文本为空，跳过该样本
                continue
            target_text = "\n".join(target_lines)
            # target_text = target_text + "\n<stop>"
            # out = self.truncate_prompt_only_if_needed(prompt_text, target_text)
            # if out is None:
            #     continue
            # truncated_prompt, truncated_target = out

            sample_dict = {
                "video_index": vidx,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "future_frames": future_frames,
                "observed_text": observed_text,
                "observed_anno": observed_anno
            }
            if self.ag.script_require:
                sample_dict["script"] = script
            if self.include_timestamps:
                sample_dict["include_timestamps"] = True

            ### MOD START: 尾类过采样
            # 判断 target 是否包含 tail class
            # target_objs = set()
            # for line in target_lines:
            #     if ':' in line:
            #         target_objs.update([o.strip()
            #                             for o in line.split(':', 1)[1].split(',')])
            # has_tail = bool(target_objs & self.tail_set)
            # dup_times = TAIL_DUP_FACTOR if has_tail else 1
            # for _ in range(dup_times):
            #     self.samples.append(sample_dict.copy())
            ### MOD END
            self.samples.append(sample_dict)
        print(f"[AGForLLM_ObjectPrediction] Total samples built: {len(self.samples)}")
    
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
            else:
                truncated_prompt_lines.pop(0)
        return None

    def _compare_frame_data(self, frame_a, frame_b):
        """比较两帧是否具有相同的scene graph"""
        if len(frame_a) != len(frame_b):
            return False
        for idx in range(1, len(frame_a)):
            obj_a = frame_a[idx]
            obj_b = frame_b[idx]
            if obj_a.get('class', -1) != obj_b.get('class', -1):
                return False
            attn_a = sorted(obj_a.get('attention_relationship', []))
            attn_b = sorted(obj_b.get('attention_relationship', []))
            if attn_a != attn_b:
                return False
            spat_a = sorted(obj_a.get('spatial_relationship', []))
            spat_b = sorted(obj_b.get('spatial_relationship', []))
            if spat_a != spat_b:
                return False
            cont_a = sorted(obj_a.get('contacting_relationship', []))
            cont_b = sorted(obj_b.get('contacting_relationship', []))
            if cont_a != cont_b:
                return False
        return True

    def _aggregate_scene_graph(self, frames, include_timestamps=False):
        """聚合帧的scene graph，合并相同帧并标注Frame ID区间"""
        intervals = []
        start_idx = 0
        n = len(frames)
        fps = 24  # 假设24帧每秒
        
        while start_idx < n:
            end_idx = start_idx
            while end_idx + 1 < n and self._compare_frame_data(frames[end_idx], frames[end_idx + 1]):
                end_idx += 1
            frame_start = self._extract_frame_number(frames[start_idx][0].get('frame', '0'))
            frame_end = self._extract_frame_number(frames[end_idx][0].get('frame', '0'))
            intervals.append((frame_start, frame_end, frames[start_idx]))
            start_idx = end_idx + 1

        all_lines = []
        for (fr_s, fr_e, frame_data) in intervals:
            if include_timestamps:
                # 添加时间戳
                time_start = time_encoding(fr_s, fps)
                time_end = time_encoding(fr_e, fps)
                
                if fr_s == fr_e:
                    all_lines.append(f"Frame {fr_s} [T={time_start['absolute_time']:.2f}s]:")
                else:
                    all_lines.append(f"Frame {fr_s}-{fr_e} [T={time_start['absolute_time']:.2f}s-{time_end['absolute_time']:.2f}s]:")
            else:
                # 原始格式
                if fr_s == fr_e:
                    all_lines.append(f"Frame {fr_s}:")
                else:
                    all_lines.append(f"Frame {fr_s}-{fr_e}:")
            for obj in frame_data[1:]:
                cls_idx = obj.get('class', -1)
                obj_name = self.object_classes[cls_idx] if 0 <= cls_idx < len(self.object_classes) else "unknown"
                attn_ids = obj.get('attention_relationship', [])
                if hasattr(attn_ids, 'tolist'):
                    attn_ids = attn_ids.tolist()
                attn_str = ",".join([self.attn_rel_classes[i] for i in attn_ids]) if attn_ids else "None"
                spat_ids = obj.get('spatial_relationship', [])
                if hasattr(spat_ids, 'tolist'):
                    spat_ids = spat_ids.tolist()
                spat_str = ",".join([self.spat_rel_classes[i] for i in spat_ids]) if spat_ids else "None"
                cont_ids = obj.get('contacting_relationship', [])
                if hasattr(cont_ids, 'tolist'):
                    cont_ids = cont_ids.tolist()
                cont_str = ",".join([self.cont_rel_classes[i] for i in cont_ids]) if cont_ids else "None"
                line = f"object: {obj_name} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
                all_lines.append(line)
        return "\n".join(all_lines)

    # ---------- 新增：统计尾类出现频次 ----------
    def _build_class_frequency(self):
        freq = np.zeros(len(self.object_classes), dtype=int)
        for video in self.ag.gt_annotations:
            for frame in video:
                for obj in frame[1:]:
                    cid = obj.get('class', -1)
                    if 0 <= cid < len(freq):
                        freq[cid] += 1
        self.tail_set = {self.object_classes[i]
                         for i, c in enumerate(freq) if c <= TAIL_FREQ_THRESHOLD}

    def _extract_frame_number(self, frame_info, video_id = False):
            """从帧信息中提取帧号，如果video_id为True则返回视频ID"""
            try:
                if video_id:
                    # 提取视频ID：获取最后一个'/'之前的所有内容
                    return frame_info.rsplit('/', 1)[0].split('.')[0]
                # 提取帧号：获取最后一个'/'之后，'.'之前的数字
                return int(frame_info.split('/')[-1].split('.')[0])
            except:
                return 0

    def _build_prompt(self, observed_text, future_frames, observed_anno, max_length = 2048, script = False, include_timestamps=False, subject_id = False):
        fps = 24  # 假设24帧每秒
        format=""
        if subject_id:
            format += "You should predict the subject index and the scene name for this scenario description first, and then output the predicted objects for the future frames.\n"
        if script:
            header = (
            "You are an object prediction assistant for scene understanding. In this task, you are provided with observed "
            "scene information from past frames and a list of future frame numbers. Your task is to predict the script first and then predict the possible objects for the exact future frames and answer in a fixed format."
        )
            format += (
                "Please output in the following format:\n"
                "Script: <script>\n"
                "Objects prediction: \n"
            )  
            if include_timestamps:
                format += "Frame <index> [T=<seconds>s]: <objects>\n"
            else:
                format += "Frame <index>: <objects>\n"
            format += "Each frame should be on a separate line with no additional commentary.\n\n"
        else: 
            header = (
            "You are an object prediction assistant for scene understanding. In this task, you are provided with observed "
            "scene information from past frames and a list of future frame numbers. Your task is to predict the possible objects for the exact future frames and answer in a fixed format."
        )
            format += (
                "Please output in the following format:\n"
            )     
            if include_timestamps:
                format += "Frame <index> [T=<seconds>s]: <objects>\n"
            else:
                format += "Frame <index>: <objects>\n"
            format += "Each frame should be on a separate line with no additional commentary.\n\n"
        

        # notice= (
        #     "IMPORTANT: Objects may appear or disappear over time. Consider the following:\n"
        #     "1. Objects that were recently visible may still be present even if not mentioned\n"
        #     "2. New objects may appear as the scene changes\n"
        #     "3. Some objects may disappear from view as time progresses\n"
        #     "4. The longer the time gap, the more likely the scene has changed significantly\n")
        
        object_classes_text = "Available objects: " + ", ".join(self.object_classes) + "\n\n"

        # 为未来帧添加时间戳
        if include_timestamps:
            future_frames_with_time = []
            for frame in future_frames:
                time_info = time_encoding(frame, fps)
                future_frames_with_time.append(f"Frame {frame} [T={time_info['absolute_time']:.2f}s]")
            frames_text = "Future frame numbers to predict objects for: " + ", ".join(future_frames_with_time) + "\n"
        else:
            frames_text = "Future frame numbers to predict objects for: " + ", ".join(map(str, future_frames)) + "\n"

        prompt = header + object_classes_text + observed_text + "\n" + format + frames_text
        new_prompt = self.truncate_prompt_if_needed(prompt, max_length, time = include_timestamps)
        return new_prompt

    
    def truncate_prompt_if_needed(self, prompt, max_length=1024, time = False):
        """
        检查 prompt 的 token 长度是否超过 max_length。如果超过，则截取 observed_text，
        通过移除较早的 frame，直到 token 长度小于或等于 max_length。

        Args:
            prompt (str): 完整的 prompt 字符串。
            max_length (int): 最大允许的 token 长度，默认为 1024。

        Returns:
            str: 可能经过截取的 prompt。

        Raises:
            ValueError: 如果移除所有 frame 后 prompt 仍超过 max_length。
        """
        # 对 prompt 进行 tokenization，计算当前 token 长度
        tokens = self.tokenizer.encode(prompt)
        current_length = len(tokens)

        # 如果 token 长度未超过限制，直接返回原 prompt
        if current_length <= max_length:
            return prompt

        # 提取 observed_text 部分
        # prompt 的结构为：header + object_classes_text + observed_text + "\n" + format + frames_text
        start_idx = len("You are an object prediction assistant for scene understanding. In this task, you are provided with observed scene information from past frames and a list of future frame numbers. Your task is to predict the possible objects for the exact future frames and answer in a fixed format.") + len("Available objects: ") + len(", ".join(self.object_classes)) + 2  # +2 表示 "\n\n"
        end_idx = prompt.find("\nPlease output in the following format:")
        observed_text = prompt[start_idx:end_idx].strip()

        # 使用正则表达式解析 observed_text 中的 frame 部分
        if time:
            frame_pattern = r'(Frame \d+(?:-\d+)?(?:\s+\[T=[\d\.]+s(?:-[\d\.]+s)?\])?:.*?)(?=Frame \d+(?:-\d+)?(?:\s+\[T=[\d\.]+s(?:-[\d\.]+s)?\])?:|$)'
        else:
            frame_pattern = r'(Frame \d+:.*?)(?=Frame \d+:|$)'  # 匹配每个 Frame，直到下一个 Frame 或结束
        frame_sections = re.findall(frame_pattern, observed_text, re.DOTALL)

        # 如果没有 frame 可截取，且仍超长，则抛出异常
        if not frame_sections:
            raise ValueError("Prompt 超过 max_length，且无法进一步截取。")

        # 逐步移除最早的 frame，直到 token 长度符合要求
        while current_length > max_length and frame_sections:
            frame_sections.pop(0)  # 移除最早的 frame
            new_observed_text = "".join(frame_sections).strip()
            new_prompt = prompt[:start_idx] + new_observed_text + prompt[end_idx:]
            tokens = self.tokenizer.encode(new_prompt)
            current_length = len(tokens)
        # print(f"Befor truncation: {prompt}, After truncation: {new_prompt}")

        # 如果移除所有 frame 后仍超长，抛出异常
        if current_length > max_length:
            raise ValueError("即使移除所有 frame，Prompt 仍超过 max_length。")

        return new_prompt

    # 以下方法保持不变，但为完整性保留
    def _group_segments_by_object(self, segments):
        obj_dict = defaultdict(list)
        for seg in segments:
            obj_cls = seg["object_class"]
            obj_dict[obj_cls].append(seg)
        for cls_ in obj_dict:
            obj_dict[cls_].sort(key=lambda x: x["start_time"])
        return dict(obj_dict)

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

    def _construct_segment_text(self, start_time, end_time, seg, obj_cls, include_time=False, add_obj_marker=True, ignore_obj_mode=False):
        attn_str = ",".join([self.attn_rel_classes[id_] for id_ in seg["attn_ids"]]) or "None"
        spat_str = ",".join([self.spat_rel_classes[id_] for id_ in seg["spat_ids"]]) or "None"
        cont_str = ",".join([self.cont_rel_classes[id_] for id_ in seg["cont_ids"]]) or "None"
        if start_time < end_time:
            time_text = f"Frame {start_time}..{end_time}: " if include_time else ""
        else:
            time_text = f"Frame {end_time}: " if include_time else ""
        obj_text = f"object: {obj_cls}" if add_obj_marker else f"Object[{obj_cls}]"
        text = f"{time_text}{obj_text} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
        return text
    
    def _generate_trend_description(self, observed_anno):
        object_trends = {}
        for frame_data in observed_anno:
            frame_num = self._extract_frame_number(frame_data[0]['frame'])
            objects_in_frame = [self.object_classes[obj['class']] for obj in frame_data[1:] if 0 <= obj['class'] < len(self.object_classes)]
            for obj in objects_in_frame:
                if obj not in object_trends:
                    object_trends[obj] = []
                object_trends[obj].append(frame_num)
        trend_description = "Observed trends:\n"
        for obj, frames in object_trends.items():
            if len(frames) > 1:
                trend = "increasing" if frames[-1] > frames[0] else "decreasing"
                trend_description += f"- {obj}: {trend} appearance over time.\n"
        return trend_description if trend_description != "Observed trends:\n" else "Observed trends: No significant trends detected.\n"


class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # 检查生成序列的最后一个 token 是否为停止符
        # 这里假设 batch size 为1
        if input_ids[0][-1].item() == self.stop_token_id:
            return True
        return False
#################################################
# 2. 模型类：SceneGraphAllocator
#################################################
class SceneGraphAllocator:
    def __init__(self, model_path, local_rank, world_size, phase="train", lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 learning_rate=1e-4, epochs=3, max_seq_length=2048, gradient_accumulation_steps=4,
                 ckpt_path=None, object_classes=None, label_smoothing = False, len=5, beta=0.5):
        self.local_rank = local_rank
        self.world_size = world_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ckpt_path = ckpt_path
        self.phase = phase  # 设置模式
        self.object_classes = object_classes
        self.label_smoothing = label_smoothing
        self.len = len
        self.beta = beta

        self.device = torch.device("cuda", self.local_rank)

        # 1) 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 添加特殊标记
        special_tokens = {"additional_special_tokens": ["<obj>", "<stop>"]}
        # special_tokens = {"additional_special_tokens": ["<obj>"]}
        self.tokenizer.add_special_tokens(special_tokens)

        # 2) 选择训练模式或推理模式
        if self.phase == "eval":
            self._initialize_for_evaluation(model_path)
        else:
            self._initialize_for_training(model_path, lora_r, lora_alpha, lora_dropout)

            # # 3) 如果提供 checkpoint，加载
            # if self.ckpt_path:
            #     self.load_checkpoint(self.ckpt_path)
        
        self.obj_line_ranges = None
        # 获取停止符 token 的 id，确保 "<stop>" 已加入特殊token中
        stop_token_id = self.tokenizer.encode("<stop>", add_special_tokens=False)[0]
        self.stopping_criteria = StoppingCriteriaList([StopOnTokenCriteria(stop_token_id=stop_token_id)])

    def _initialize_for_evaluation(self, model_path):
        """初始化用于推理的模型"""
        print("[Info] Initializing model in EVAL mode...")

        # 加载基础 CausalLM 模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": f"cuda:{self.local_rank}"} if self.world_size > 1 else "auto"
        ).to(self.device)
        base_model.resize_token_embeddings(len(self.tokenizer))

        # 加载 LoRA
        peft_model = PeftModel.from_pretrained(base_model, self.ckpt_path).to(self.device)
        peft_model.eval()
        for p in peft_model.parameters():
            p.requires_grad_(False)

        self.model = peft_model



    def _initialize_for_training(self, model_path, lora_r, lora_alpha, lora_dropout):
        """初始化用于训练的模型"""
        print("[Info] Initializing model in TRAIN mode...")

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            # torch_dtype=torch.float16,
            # device_map={"": f"cuda:{self.local_rank}"} if self.world_size > 1 else "auto"
        ).to(self.device)
        base_model.resize_token_embeddings(len(self.tokenizer))

        # 配置 LoRA
        if self.ckpt_path is None:
            # 配置 LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
                layers_to_transform=list(range(4, 28, 4)),
            )
            peft_model = get_peft_model(base_model, lora_config)
            peft_model.train().to(self.device)
        else:
            # 从检查点加载模型
            peft_model = PeftModel.from_pretrained(
                base_model,
                self.ckpt_path
            ).to(self.device)
            print("✓ LoRA 权重加载成功")

            for name, param in peft_model.named_parameters():
                if "lora" in name:  # 只对LoRA参数启用梯度
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        peft_model.train().to(self.device)
        self.model = peft_model

        # 调整 Tokenizer

    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"Loading checkpoint from {checkpoint_path} ...")
        try:
            # 1. 加载 LoRA 权重
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                safe_serialization=True,
                torch_dtype=torch.float32
            ).to(self.device)

            # 2. 确保模型在训练模式下
            self.model.train()
            
            # 3. 确保参数需要梯度
            for param in self.model.parameters():
                param.requires_grad = True

            # 4. 验证
            has_params_need_grad = any(p.requires_grad for p in self.model.parameters())
            print(f"[Debug] After loading checkpoint, model has parameters requiring gradients: {has_params_need_grad}")
            
            print("[Info] LoRA adapter loaded successfully.")
        except Exception as e:
            print(f"[Warning] Failed to load LoRA adapter: {e}")
    
    def set_obj_line_ranges(self, obj_line_ranges):
        """设置 obj_line_ranges，用于 Scene Graph 分配"""
        self.obj_line_ranges = obj_line_ranges
    
    def assign_scene_graphs(self, observed_segments, lines_batch, obj_list, future_frames, observed_anno, device=None):
        """
        为每个对象的每个未来帧分配 Scene Graph。
        
        参数:
            lines_batch (list): 包含所有对象的 Scene Graph 行的列表
            obj_list (list): 对象类别列表
            future_frames (list): 需要分配的未来帧编号列表
        
        返回:
            list: 分配后的 Scene Graph 列表
        """
        assigned_scene_graphs = []
        for i_obj, obj_cls in enumerate(obj_list):
            prompt = self.build_prompt(observed_segments, future_frames, observed_anno, max_length=self.max_seq_length, include_timestamps=True)
            generated_text = self.generate_text(prompt, device=device)
            assigned_sg_for_obj = self.parse_generated_text(generated_text, lines_batch, future_frames)
            assigned_scene_graphs.extend(assigned_sg_for_obj)
        return assigned_scene_graphs

    def build_prompt(self, observed_text, future_frames, observed_anno, max_length = 2048, script = False, include_timestamps=False, subject_id = False):
        fps = 24  # 假设24帧每秒
        format=""
        if subject_id:
            format += "You should predict the subject index and the scene name for this scenario description first, and then output the predicted objects for the future frames.\n"
        if script:
            header = (
            "You are an object prediction assistant for scene understanding. In this task, you are provided with observed "
            "scene information from past frames and a list of future frame numbers. Your task is to predict the script first and then predict the possible objects for the exact future frames and answer in a fixed format."
        )
            format += (
                "Please output in the following format:\n"
                "Script: <script>\n"
                "Objects prediction: \n"
            )  
            if include_timestamps:
                format += "Frame <index> [T=<seconds>s]: <objects>\n"
            else:
                format += "Frame <index>: <objects>\n"
            format += "Each frame should be on a separate line with no additional commentary.\n\n"
        else: 
            header = (
            "You are an object prediction assistant for scene understanding. In this task, you are provided with observed "
            "scene information from past frames and a list of future frame numbers. Your task is to predict the possible objects for the exact future frames and answer in a fixed format."
        )
            format += (
                "Please output in the following format:\n"
            )     
            if include_timestamps:
                format += "Frame <index> [T=<seconds>s]: <objects>\n"
            else:
                format += "Frame <index>: <objects>\n"
            format += "Each frame should be on a separate line with no additional commentary.\n\n"
        

        notice= (
            "IMPORTANT: Objects may appear or disappear over time. Consider the following:\n"
            "1. Objects that were recently visible may still be present even if not mentioned\n"
            "2. New objects may appear as the scene changes\n"
            "3. Some objects may disappear from view as time progresses\n"
            "4. The longer the time gap, the more likely the scene has changed significantly\n")
        
        object_classes_text = "Available objects: " + ", ".join(self.object_classes) + "\n\n"

        # 为未来帧添加时间戳
        if include_timestamps:
            future_frames_with_time = []
            for frame in future_frames:
                time_info = time_encoding(frame, fps)
                future_frames_with_time.append(f"Frame {frame} [T={time_info['absolute_time']:.2f}s]")
            frames_text = "Future frame numbers to predict objects for: " + ", ".join(future_frames_with_time) + "\n"
        else:
            frames_text = "Future frame numbers to predict objects for: " + ", ".join(map(str, future_frames)) + "\n"

        prompt = header + object_classes_text + observed_text + "\n" + format + frames_text
        # new_prompt = self.truncate_prompt_if_needed(prompt, max_length, time = include_timestamps)
        return prompt


    def truncate_prompt_if_needed(self, prompt, max_length=1024, time = False):
        """
        检查 prompt 的 token 长度是否超过 max_length。如果超过，则截取 observed_text，
        通过移除较早的 frame，直到 token 长度小于或等于 max_length。

        Args:
            prompt (str): 完整的 prompt 字符串。
            max_length (int): 最大允许的 token 长度，默认为 1024。

        Returns:
            str: 可能经过截取的 prompt。

        Raises:
            ValueError: 如果移除所有 frame 后 prompt 仍超过 max_length。
        """
        # 对 prompt 进行 tokenization，计算当前 token 长度
        tokens = self.tokenizer.encode(prompt)
        current_length = len(tokens)

        # 如果 token 长度未超过限制，直接返回原 prompt
        if current_length <= max_length:
            return prompt

        # 提取 observed_text 部分
        # prompt 的结构为：header + object_classes_text + observed_text + "\n" + format + frames_text
        start_idx = len("You are an object prediction assistant for scene understanding. In this task, you are provided with observed scene information from past frames and a list of future frame numbers. Your task is to predict the possible objects for the exact future frames and answer in a fixed format.") + len("Available objects: ") + len(", ".join(self.object_classes)) + 2  # +2 表示 "\n\n"
        end_idx = prompt.find("\nPlease output in the following format:")
        observed_text = prompt[start_idx:end_idx].strip()

        # 使用正则表达式解析 observed_text 中的 frame 部分
        if time:
            frame_pattern = r'(Frame \d+(?:-\d+)?(?:\s+\[T=[\d\.]+s(?:-[\d\.]+s)?\])?:.*?)(?=Frame \d+(?:-\d+)?(?:\s+\[T=[\d\.]+s(?:-[\d\.]+s)?\])?:|$)'
        else:
            frame_pattern = r'(Frame \d+:.*?)(?=Frame \d+:|$)'  # 匹配每个 Frame，直到下一个 Frame 或结束
        frame_sections = re.findall(frame_pattern, observed_text, re.DOTALL)

        # 如果没有 frame 可截取，且仍超长，则抛出异常
        if not frame_sections:
            raise ValueError("Prompt 超过 max_length，且无法进一步截取。")

        # 逐步移除最早的 frame，直到 token 长度符合要求
        while current_length > max_length and frame_sections:
            frame_sections.pop(0)  # 移除最早的 frame
            new_observed_text = "".join(frame_sections).strip()
            new_prompt = prompt[:start_idx] + new_observed_text + prompt[end_idx:]
            tokens = self.tokenizer.encode(new_prompt)
            current_length = len(tokens)
        # print(f"Befor truncation: {prompt}, After truncation: {new_prompt}")

        # 如果移除所有 frame 后仍超长，抛出异常
        if current_length > max_length:
            raise ValueError("即使移除所有 frame，Prompt 仍超过 max_length。")

        return new_prompt
    
    def collate_fn_eval(self, batch):
        """Evaluation-specific collate function"""
        sample = batch[0]  # Batch size is 1 during evaluation
        script = "script" in sample and sample["script"]
        include_timestamps = "include_timestamps" in sample and sample["include_timestamps"]
        prompt_text = self.build_prompt(sample["observed_text"], sample["future_frames"], sample["observed_anno"], max_length=self.max_seq_length, script=script, include_timestamps=include_timestamps)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_length)
        return {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "future_frames": sample["future_frames"],
            "target_text": sample["target_text"],
            "prompt_text": prompt_text,
            "observed_text": sample["observed_text"],
            "include_timestamps": include_timestamps
        }
    
    def _extract_frame_number(self, frame_info):
        """从帧信息中提取帧号"""
        try:
            return int(frame_info.split('/')[-1].split('.')[0])
        except:
            return 0
        
    def _generate_trend_description(self, observed_anno):
            object_trends = {}
            for frame_data in observed_anno:
                frame_num = self._extract_frame_number(frame_data[0]['frame'])
                objects_in_frame = [self.object_classes[obj['class']] for obj in frame_data[1:] if 0 <= obj['class'] < len(self.object_classes)]
                for obj in objects_in_frame:
                    if obj not in object_trends:
                        object_trends[obj] = []
                    object_trends[obj].append(frame_num)
            trend_description = "Observed trends:\n"
            for obj, frames in object_trends.items():
                if len(frames) > 1:
                    trend = "increasing" if frames[-1] > frames[0] else "decreasing"
                    trend_description += f"- {obj}: {trend} appearance over time.\n"
            return trend_description if trend_description != "Observed trends:\n" else "Observed trends: No significant trends detected.\n"

    def generate_text(self, prompts, max_new_tokens=256, temperature=0.7, top_p=0.95, do_sample=True):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria  # 加入停止条件
            )
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    def parse_generated_objects(self, generated_text, time = False):
        """
        Parse the generated text to extract the predicted objects for each frame.
        Returns a dictionary in the form: {frame_number: [object1, object2, ...], ...}
        """
        if time:
            pattern = re.compile(r"Frame (\d+)(?:-(\d+))?(?:\s+\[T=([\d\.]+)s(?:-([\d\.]+)s)?\])?:\s*(.*)")
        else:
            pattern = re.compile(r"Frame (\d+):\s*(.*)")
        predictions = {}
        for line in generated_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                frame_num = int(match.group(1))
                if time:
                    objs_str = match.group(5)
                else:
                    objs_str = match.group(2)
                objects = [obj.strip() for obj in objs_str.split(',') if obj.strip()]
                predictions[frame_num] = objects
        return predictions

    def collate_fn(self, batch):
        input_ids_list = []
        label_ids_list = []
        frame_counts = []  # 新增：记录每个样本的帧数

        for sample in batch:
            prompt_text = sample["prompt_text"]
            target_text = sample["target_text"]
            prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False)
            prompt_ids = prompt_enc["input_ids"]
            prompt_len = len(prompt_ids)
            target_enc = self.tokenizer(target_text, add_special_tokens=False)
            target_ids = target_enc["input_ids"]
            full_ids = prompt_ids + target_ids
            # if len(full_ids) > self.max_seq_length:
            #     full_ids = full_ids[:self.max_seq_length]
            labels = [-100] * prompt_len + target_ids
            labels = labels[:len(full_ids)]
            input_ids_list.append(full_ids)
            label_ids_list.append(labels)

            # 计算 target_text 中的帧数
            frame_count = target_text.count("Frame")
            frame_counts.append(frame_count)

        padded_inp = self.tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt")
        padded_label = self.tokenizer.pad({"input_ids": label_ids_list}, return_tensors="pt")["input_ids"]
        input_ids_ = padded_inp["input_ids"]
        attn_mask = (input_ids_ != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids_.to(self.device),
            "attention_mask": attn_mask.to(self.device),
            "labels": padded_label.to(self.device),
            "future_frames": [sample["future_frames"] for sample in batch],
            "prompt_texts": [sample["prompt_text"] for sample in batch],
            "target_texts": [sample["target_text"] for sample in batch],
            "observed_text": [sample["observed_text"] for sample in batch],
            "frame_counts": frame_counts  # 新增：返回帧数
        }

    def train_loop(self, dataset, batch_size=2, warmup_steps=1000, save_path="./allocator_finetuned"):
        from torch.nn.parallel import DistributedDataParallel as DDP
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True
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
        optimizer = AdamW(self.ddp_model.parameters(), lr=self.learning_rate)
        total_steps = (len(loader) // self.gradient_accumulation_steps) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        global_step = 0
        for epoch in range(self.epochs):
            sampler.set_epoch(epoch)
            for step, batch_data in enumerate(tqdm(loader, desc=f"[DDP] Ep{epoch+1} rank={self.local_rank}")):
                outputs = self.ddp_model(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    labels=batch_data["labels"],
                    output_hidden_states=True,
                    return_dict=True
                )
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                labels = batch_data["labels"]  # [batch_size, seq_len]
                frame_counts = batch_data["frame_counts"]  # List of int

                loss = self.compute_weighted_ce_loss_debug(logits, labels, lent=self.len)
                # loss=outputs.loss
                if torch.isnan(loss):
                    print(f"[Warning] NaN loss detected at epoch {epoch+1}, step {step}")
                    self._debug_batch(batch_data, outputs)
                    breakpoint()
                    continue  # 跳过当前批次，避免 nan 影响后续计算
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=0.5)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % 50 == 0 and self.local_rank == 0:
                        print(f"[Rank0] ep={epoch+1}, step={global_step} | loss={loss.item()*self.gradient_accumulation_steps:.4f}")
            if self.local_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                # 总是保存为 checkpoint
                # checkpoint_dir = os.path.join(save_path, "checkpoint")
                # self.save_checkpoint(checkpoint_dir, optimizer, scheduler)
                # print(f"[Rank0] epoch {epoch+1} saved => {checkpoint_dir}")
                
                # 如果是 5 的倍数，额外保存一个带编号的里程碑检查点
                if (epoch + 1) % 5 == 0:
                    epoch_dir = os.path.join(save_path, f"epoch_{epoch+1}")
                    self.save_checkpoint(epoch_dir, optimizer, scheduler)
                    print(f"[Rank0] epoch {epoch+1} milestone saved => {epoch_dir}")
        if self.world_size > 1:
            dist.destroy_process_group()

    def compute_weighted_ce_loss_debug_2(self, logits, labels, lent=10):
        # 对 logits 和 labels 进行移位：忽略第一个 token 的预测，使用后续 token 进行预测
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        batch_size = shift_logits.size(0)
        seq_len = shift_logits.size(1)
        total_loss = 0.0
        # 获取 "Frame" token 的 id（假设它只对应一个 token）
        frame_token_id = self.tokenizer.encode("Frame", add_special_tokens=False)[0]

        # 定义基于 cosine 的权重函数：t in [0,1] -> weight in [1.5, 0.5]
        def weight_func(t):
            # 当 t=0 时: cos(0)=1, 得到 0.5*(1+1)+0.5 = 1.5；
            # 当 t=1 时: cos(pi)= -1, 得到 0.5*(1-1)+0.5 = 0.5；
            return self.beta * (1 + math.cos(math.pi * t)) + 1 - self.beta
        
        for b in range(batch_size):
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            # 计算移位后每个 token 的 loss，形状：[seq_len]
            sample_loss = loss_fn(shift_logits[b].contiguous().view(-1, shift_logits.size(-1)),
                                    shift_labels[b].contiguous().view(-1))
            sample_loss = sample_loss.view(seq_len)
            # mask 有效 token
            mask = (shift_labels[b] != -100).float().to(self.device)
            
            # 获取当前样本中所有出现 "Frame" 的位置
            frame_positions = (shift_labels[b] == frame_token_id).nonzero(as_tuple=True)[0]
            if len(frame_positions) > lent:
                # 取第 lent 个 Frame 的位置作为阈值
                threshold_pos = frame_positions[lent-1].item()
                # 初始化权重向量全为 1
                weights = torch.ones(seq_len, dtype=torch.float32, device=self.device)
                # 设定归一化区间：[threshold_pos+1, target_end)
                target_end = int(mask.sum().item())  # 有效 token 数量
                denom = target_end - (threshold_pos + 1)
                if denom <= 0:
                    denom = 1  # 避免除0
                for i in range(threshold_pos + 1, target_end):
                    t_norm = (i - (threshold_pos + 1)) / denom  # t_norm 在 [0,1]
                    weights[i] = weight_func(t_norm)
                # 对有效 token 的 loss 做加权平均
                denominator = (weights * mask.view(-1)).sum() + 1e-10
                weighted_loss = (sample_loss * weights * mask.view(-1)).sum() / denominator
                total_loss += weighted_loss
            else:
                # 如果没有足够的 Frame 信息，直接使用 unweighted loss
                weighted_loss = (sample_loss * mask.view(-1)).sum() / mask.sum()
                total_loss += weighted_loss
        return total_loss / batch_size
    
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
            mask = (shift_labels[b] != -100).float().to(self.device)
            
            # 获取当前样本中所有出现 "Frame" 的位置
            frame_positions = (shift_labels[b] == frame_token_id).nonzero(as_tuple=True)[0]
            if len(frame_positions) > lent:
                # 第 lent 个 Frame 的位置（索引从 0 开始）
                fifth_frame_pos = frame_positions[lent-1].item()
                # 初始化权重全1
                weights = torch.ones(seq_len, dtype=torch.float32, device=self.device)
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

    def compute_weighted_ce_loss(self, logits, labels, frame_counts, lent = 5):
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        total_loss = 0.0
        frame_token_id = self.tokenizer.encode("Frame", add_special_tokens=False)[0]  # 获取 "Frame" 的 token ID

        for b in range(batch_size):
            frame_count = frame_counts[b]
            # 找到 target_text 的起始位置（labels 从 -100 变为有效 token）
            target_start = (labels[b] != -100).nonzero(as_tuple=True)[0][0].item()
            target_len = (labels[b] != -100).sum().item()
            target_end = target_start + target_len

            if frame_count <= lent:
                # 帧数 <= 5，不加权
                if self.label_smoothing:
                        loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100, label_smoothing=0.1)
                else:
                    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                loss = loss_fn(logits[b].view(-1, logits.size(-1)), labels[b].view(-1))
                total_loss += loss.mean()
            else:
                # 帧数 > 5，加权从第 5 帧之后的 token 开始
                # 统计 labels 中 Frame 的位置
                frame_positions = (labels[b] == frame_token_id).nonzero(as_tuple=True)[0]
                if len(frame_positions) > lent:
                    fifth_frame_pos = frame_positions[lent-1].item()  # 第 5 个 Frame 的位置
                    # 从第 5 帧之后的 token 开始加权
                    weights = torch.ones(seq_len, dtype=torch.float32).to(self.device)
                    for i in range(fifth_frame_pos + 1, target_end):
                        if labels[b, i] != -100:
                            # sigmoid 权重：缓慢减小到快速减小
                            t = (i - fifth_frame_pos) / (target_end - fifth_frame_pos)
                            weights[i] = 1.0-(1.0 / (1.0 + math.exp(-1 * (t - 0.5))))  # sigmoid 曲线
                    if self.label_smoothing:
                        loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100, label_smoothing=0.1)
                    else:
                        loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

                    mask = (labels[b] != -100).float().to(self.device)
                    loss = loss_fn(logits[b].view(-1, logits.size(-1)), labels[b].view(-1))
                    total_loss += (loss * mask.view(-1)).sum() / mask.sum()
                    # weighted_loss = (loss * weights * mask).sum() / (weights * mask).sum()
                    # total_loss += weighted_loss
                else:
                    # Frame 少于 5 个，不加权
                    if self.label_smoothing:
                        loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100, label_smoothing=0.1)
                    else:
                        loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    mask = (labels[b] != -100).float().to(self.device)
                    loss = loss_fn(logits[b].view(-1, logits.size(-1)), labels[b].view(-1))
                    total_loss += (loss * mask.view(-1)).sum() / mask.sum()

        return total_loss / batch_size

    def save_checkpoint(self, checkpoint_path, optimizer, scheduler):
        os.makedirs(checkpoint_path, exist_ok=True)
        self.ddp_model.module.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(checkpoint_path, "training_state.pt"))
        print(f"Checkpoint saved at {checkpoint_path}")

    def _debug_batch(self, batch_data, outputs):
        """调试模式：打印批次信息以分析 loss=nan 的原因"""
        print("[Debug] Batch Data Info:")
        print(f"  - prompt_text: {batch_data['prompt_texts']}")
        print(f"  - target_text: {batch_data['target_texts']}")
        print(f"  - future_frames: {batch_data['future_frames']}")
        print(f"  - observed_text: {batch_data['observed_text']}")
        print("[Debug] Model Outputs Info:")
        print(f"  - logits shape: {outputs.logits.shape}")
        print(f"  - labels shape: {batch_data['labels'].shape}")
        print(f"  - loss: {outputs.loss}")
        # 可选：打印部分 logits 和 labels 的具体值以进一步检查
        print(f"  - sample logits: {outputs.logits[0, :5]}")
        print(f"  - sample labels: {batch_data['labels'][0, :5]}")

    def evaluate(self, eval_dataset, path=None, temperature=0.7, top_p=0.4):    
        """
        评估模型性能并可选择保存预测结果
        
        Args:
            eval_dataset: 评估数据集
            path: 保存预测结果的JSON文件路径,如果为None则不保存
        """
        if self.local_rank != 0:
            return
            
        self.eval_dataset = eval_dataset
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn_eval
        )
        
        self.model.eval()
        total_frames = 0
        total_correct = 0
        total_predictions = 0
        strict_correct = 0
        pred_contains_gt = 0
        partial_overlap = 0
        no_overlap = 0
        num = 0
        
        # 用于记录预测结果的字典
        results = {
        "videos": {},  # 修改：添加 videos 键
    }
        
        with torch.no_grad():
            for video_id, batch in enumerate(tqdm(eval_loader, desc="Evaluating"), 1):
                future_frames = batch['future_frames']
                target_text = batch['target_text']
                prompt_text = batch['prompt_text']

                # 生成预测
                generated_text_raw = self.generate_text(prompt_text, max_new_tokens=420, temperature=temperature, top_p=top_p)[0]
                generated_text = generated_text_raw.replace(prompt_text, "")
                pred_objects = self.parse_generated_objects(generated_text, time=batch["include_timestamps"])
                gt_objects = self.parse_generated_objects(target_text, time=batch["include_timestamps"])
                
                # 如果需要保存结果
                if path is not None:
                    # 为每一帧创建记录
                    frame_results = []
                    for frame in future_frames:
                        frame_result = {
                            "frame_id": int(frame),
                            "prompt": prompt_text,
                            "predicted_objects": list(pred_objects.get(frame, [])),
                            "ground_truth_objects": list(gt_objects.get(frame, [])),
                            "correct_predictions": list(set(pred_objects.get(frame, [])) & set(gt_objects.get(frame, [])))
                        }
                        frame_results.append(frame_result)
                    # 将帧结果添加到对应的视频ID下
                    results["videos"][str(video_id)] = {
                        "frames": frame_results
                    }
                    # 添加到总结果中
                    # results["predictions"].extend(frame_results)

                # 评估指标计算
                for i, frame in enumerate(future_frames):
                    total_frames += 1
                    pred_objs = set(pred_objects.get(frame, []))
                    gt_objs = set(gt_objects.get(frame, []))
                    correct_objs = pred_objs & gt_objs
                    total_correct += len(correct_objs)
                    total_predictions += len(gt_objs)

                    if pred_objs == gt_objs:
                        strict_correct += 1
                    elif gt_objs <= pred_objs:
                        pred_contains_gt += 1
                    elif pred_objs & gt_objs and gt_objs - pred_objs:
                        partial_overlap += 1
                    else:
                        no_overlap += 1
                
                num += 1
                if num > 100:
                    break
        
        # 如果指定了保存路径,保存结果
        if path is not None:
            import json
            import os
            
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # path + "predictions.json"
            path = os.path.join(path, "predictions.json")
            
            # 保存结果到JSON文件
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Prediction results saved to {path}")

        # 计算并打印评估指标
        partial_accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        strict_accuracy = strict_correct / total_frames if total_frames > 0 else 0.0
        pred_contains_gt_ratio = pred_contains_gt / total_frames if total_frames > 0 else 0.0
        partial_overlap_ratio = partial_overlap / total_frames if total_frames > 0 else 0.0
        no_overlap_ratio = no_overlap / total_frames if total_frames > 0 else 0.0

        print(f"Evaluation Results:")
        print(f"Total Frames: {total_frames}")
        print(f"Partial Match Metrics:")
        print(f"  Total Predictions: {total_predictions}")
        print(f"  Correct Predictions: {total_correct}")
        print(f"  Partial Accuracy: {partial_accuracy:.4f}")
        print(f"Strict Match Metrics:")
        print(f"  Strict Correct Frames: {strict_correct}")
        print(f"  Strict Accuracy: {strict_accuracy:.4f}")
        print(f"Additional Metrics:")
        print(f"  Frames where pred_objs contains gt_objs: {pred_contains_gt} ({pred_contains_gt_ratio:.4f})")
        print(f"  Frames with partial overlap: {partial_overlap} ({partial_overlap_ratio:.4f})")
        print(f"  Frames with no overlap: {no_overlap} ({no_overlap_ratio:.4f})")

#################################################
# 3. main 函数
#################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='AG dataset root path')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--datasize', type=str, default='full')
    parser.add_argument('--script_require', action='store_true')
    parser.add_argument('--context_fraction', type=float, default=0.9)
    parser.add_argument('--llama_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./allocator_finetuned_ddp')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--temprature', type=float, default=0.7)
    parser.add_argument('--topk', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--relevance', action='store_true')
    parser.add_argument('--include_timestamps', action='store_true')
    parser.add_argument('--user_id', action='store_true')
    parser.add_argument('--len', type=int, default=5) #verify, relevance
    parser.add_argument('--beta', type=float, default=1e-4)
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

    if args.evaluate:
        ag_dataset = AG(
            phase='test',
            datasize='mini',
            data_path=args.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
            script_require=args.script_require,
            verify = args.verify,
            relevance=args.relevance,
            subject_id=args.user_id
        )
    else:
        ag_dataset = AG(
            phase=args.phase,
            datasize=args.datasize,
            data_path=args.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
            script_require=args.script_require,
            verify = args.verify,
            relevance=args.relevance,
            subject_id=args.user_id
        )

    dataset_for_llm = AGForLLM_ObjectPrediction(
        ag_dataset=ag_dataset,
        context_fraction=args.context_fraction,
        max_len=args.max_seq_length,
        path=args.llama_path,
        include_timestamps=args.include_timestamps,
        subject_id=args.user_id
    )
    allocator = SceneGraphAllocator(
        model_path=args.llama_path,
        local_rank=args.local_rank,
        world_size=world_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ckpt_path=args.ckpt_path,
        phase=args.phase,
        object_classes=dataset_for_llm.object_classes,
        label_smoothing=args.label_smoothing,
        len = args.len,
        beta = args.beta
    )
    if args.evaluate:
        allocator.evaluate(dataset_for_llm, path=args.save_path, temperature=args.temprature, top_p=args.topk)
    else:
        allocator.train_loop(
            dataset_for_llm,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            save_path=args.save_path
        )
    # if world_size > 1:
    #     dist.destroy_process_group()
    if rank == 0:
        print("Done second stage training/evaluation.")

if __name__ == "__main__":
    main()