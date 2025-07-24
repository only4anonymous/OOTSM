import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import tiktoken
# 以下import根据你项目路径修改
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint
from collections import defaultdict
# 你已有的 STTran 实现
from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierTransformer, GetBoxes
from lib.word_vectors import obj_edge_vectors
import time
import copy
import numpy as np
from llama_SGA.SGA_stage_1 import SceneGraphFineTuner
from llama_SGA.SGA_stage_2 import SceneGraphAllocator
from llama_SGA.SGA_stage_0 import SceneGraphAllocator as SceneGraphFineTuner0
#####################################
# 这里是关系标签集
#####################################
REL_CLASSES = [
    'looking_at', 'not_looking_at', 'unsure',
    'above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in',
    'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on'
]
NUM_REL_CLASSES = len(REL_CLASSES)
ATTN_REL_CLASSES = ['looking_at', 'not_looking_at', 'unsure']
SPAT_REL_CLASSES = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
CONT_REL_CLASSES = ['carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship', 'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']
# _PATTERN_LINE = re.compile(
#     r'object:\s*(.*?)\s*Person attention to [^:]+:\s*([^,]+),\s*.*?\s*located relative to person:\s*([^,]+),\s*Person contact with [^:]+:\s*([^,\.]+)',
#     re.IGNORECASE
# )
# _PATTERN_LINE = re.compile(
#     r'object:\s*([^,]+?)\s*attention:\s*([^,]*?(?:,[^,]*?)*?)(?=,\s*spatial:),\s*spatial:\s*([^,]*?(?:,[^,]*?)*?)(?=,\s*contact:),\s*contact:\s*([^,]*?(?:,[^,]*?)*?)\.?$',
#     re.IGNORECASE
# )
_PATTERN_LINE = re.compile(
    r'.*?attention:\s*([^,]*?(?:,[^,]*?)*?)(?=,\s*spatial:),\s*spatial:\s*([^,]*?(?:,[^,]*?)*?)(?=,\s*contact:),\s*contact:\s*([^,]*?(?:,[^,]*?)*?)(?:\.|\s|$)',
    re.IGNORECASE
)
#########################################
# 1) 你的 STTran 类 (带 get_scene_graph_labels, print_indices)
#########################################
class STTran(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None,
                 rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None, script_required=False, object_required=False, relation_required=False):
        super(STTran, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.script_required = script_required
        self.object_required = object_required
        self.relation_required = relation_required
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.num_features = 1936

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
        self.obj_fc  = nn.Linear(2376, 512)
        self.vr_fc   = nn.Linear(256 * 7 * 7, 512)

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
        global_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.global_transformer = Encoder(global_encoder, num_layers=3)
        local_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.local_transformer = Encoder(local_encoder, num_layers=1)

        self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
        self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)

        if self.script_required:
            script_embedding_dim = 768
            self.script_proj = nn.Linear(script_embedding_dim, 256)

    def get_scene_graph_labels(self, obj_indices, attn_rel_indices, spaitial_rel_indices, rel_indices):
        """
        获取Scene Graph中的节点标签和边的关系标签。
        """
        object_labels = [self.obj_classes[obj_idx] for obj_idx in obj_indices]
        attn_relationship_labels = [self.rel_classes[rel_idx] for rel_idx in attn_rel_indices]
        spatial_relationship_labels = [self.rel_classes[rel_idx] for rel_idx in spaitial_rel_indices]
        contacting_relationship_labels = [self.rel_classes[rel_idx] for rel_idx in rel_indices]

        scene_graph_info = {
            "objects": object_labels,
            "attn_relationships": attn_relationship_labels,
            "spatial_relationships": spatial_relationship_labels,
            "contacting_relationships": contacting_relationship_labels
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

    def forward(self, entry, testing=False):
        entry = self.object_classifier(entry)
        # 脚本嵌入
        if self.script_required and "script_embeddings" in entry and entry["script_embeddings"] is not None:
            script_emb = entry["script_embeddings"]
            script_emb = script_emb.unsqueeze(0)
            script_proj = self.script_proj(script_emb)
        else:
            script_proj = None

        # visual part
        subj_rep = self.subj_fc(entry['features'][entry['pair_idx'][:, 0]])
        obj_rep  = self.obj_fc(entry['features'][entry['pair_idx'][:, 1]])

        if self.script_required and script_proj is not None and self.object_required:
            num_objects = subj_rep.size(0)
            script_proj_relevant = script_proj.expand(num_objects, -1)
            subj_rep = torch.cat([subj_rep, script_proj_relevant], dim=1)
            obj_rep  = torch.cat([obj_rep, script_proj_relevant], dim=1)

        entry["subj_rep_actual"] = subj_rep
        entry["obj_rep_actual"]  = obj_rep

        vr = self.union_func1(entry['union_feat']) + self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))

        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
        # semantic
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class  = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb   = self.obj_embed(subj_class)
        obj_emb    = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)

        if self.script_required and script_proj is not None and self.relation_required:
            num_rel = rel_features.size(0)
            script_proj_rel = script_proj.expand(num_rel, -1)
            rel_features = torch.cat([rel_features, script_proj_rel], dim=1)

        # Spatial-Temporal
        im_indices = entry["boxes"][ entry["pair_idx"][:, 1], 0]
        frames = []
        for l in im_indices.unique():
            frames.append(torch.where(im_indices == l)[0])
        frame_features = pad_sequence([rel_features[idx] for idx in frames], batch_first=True)
        masks = (1 - pad_sequence([torch.ones(len(idx)) for idx in frames], batch_first=True)).bool().cuda()
        rel_ = self.local_transformer(frame_features, src_key_padding_mask=masks)
        rel_features = torch.cat([rel_[i, :len(idx)] for i, idx in enumerate(frames)])

        sequences = []
        for l in obj_class.unique():
            k = torch.where(obj_class.view(-1) == l)[0]
            if len(k)>0:
                sequences.append(k)
        pos_index = []
        for idx in sequences:
            im_idx, counts = torch.unique(entry["pair_idx"][idx][:, 0].view(-1), return_counts=True, sorted=True)
            counts = counts.tolist()
            p = torch.cat([torch.LongTensor([img_id]*c) for img_id,c in zip(range(len(counts)), counts)])
            pos_index.append(p)

        sequence_features = pad_sequence([rel_features[idx] for idx in sequences], batch_first=True)
        in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]))).bool().cuda()
        masks2  = (1 - pad_sequence([torch.ones(len(idx)) for idx in sequences], batch_first=True)).bool().cuda()
        pos_index = pad_sequence(pos_index, batch_first=True) if self.mode=="sgdet" else None
        sequence_features = self.positional_encoder(sequence_features, pos_index)
        out = self.global_transformer(sequence_features, src_key_padding_mask=masks2, mask=in_mask)

        rel_flat = torch.cat([out[i, :len(idx)] for i, idx in enumerate(sequences)])
        indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, rel_features.shape[1])
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

#################################
# SceneSayerODE - 替换 ODE为 LLM
#################################
class SceneSayerODE(nn.Module):
    def __init__(self, mode, 
                 attention_class_num=None,
                 spatial_class_num=None, 
                 contact_class_num=None, 
                 obj_classes=None,
                 rel_classes=None,
                 enc_layer_num=None, 
                 dec_layer_num=None, 
                 max_window=None, 
                 script_required=False, 
                 object_required=False, 
                 relation_required=False,
                 use_classify_head=False,
                 llama_path=None,
                 lora_path_stage0=None,
                 lora_path_stage1=None,
                 lora_path_stage2=None,
                 use_fusion=False,
                 save_path=False):
        super(SceneSayerODE, self).__init__()
        self.mode = mode
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attn_rel_classes = ATTN_REL_CLASSES
        self.spat_rel_classes = SPAT_REL_CLASSES
        self.cont_rel_classes = CONT_REL_CLASSES
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.use_classify_head = use_classify_head

        self.relationship_categories = {
            "Attention": ATTN_REL_CLASSES,
            "Spatial": SPAT_REL_CLASSES,
            "Contact": CONT_REL_CLASSES
        }

        self.d_model = 1936
        if script_required and object_required:
            self.d_model += 768
        elif script_required:
            self.d_model += 256

        self.max_window = max_window

        # 初始化 STTran
        self.dsgdetr = STTran(
            mode=self.mode,
            obj_classes=obj_classes,
            rel_classes=rel_classes,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
            script_required=script_required,
            object_required=object_required,
            relation_required=relation_required
        )
        # 初始化 Stage 0
        # self.stage0_anticipator = SceneGraphFineTuner0(
        #     model_path=llama_path,  # llama_path
        #     ckpt_path=lora_path_stage0,  # lora_path_stage1
        #     phase="eval",
        #     local_rank=0,  # 添加默认值
        #     world_size=1,  # 添加默认值
        #     object_classes=obj_classes,
        # )

        # 初始化 Stage 1
        self.stage1_anticipator = SceneGraphFineTuner(
            model_path=llama_path,
            ckpt_path=lora_path_stage1,
            phase="eval",
            local_rank=0,  # 添加默认值
            world_size=1,  # 添加默认值
            object_classes = obj_classes,
        )

        # 初始化 Stage 2
        self.stage2_allocator = SceneGraphAllocator(
            model_path=llama_path,
            ckpt_path=lora_path_stage2,
            phase="eval",
            local_rank=0,  # 添加默认值
            world_size=1,  # 添加默认值
        )

    def forward_single_entry(self, context_fraction, entry):
        device = entry["im_idx"].device

        # Step 1: 使用 DS-GDETR 处理当前帧
        entry = self.dsgdetr(entry)
        im_idx = entry["im_idx"]
        pair_idx = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds = im_idx.size(0)
        num_frames = len(gt_annotation)

        if num_frames < 2:
            return num_frames, {}

        # Step 2: 计算观测截止帧
        end = int(torch.ceil(torch.tensor(num_frames * context_fraction)).item() - 1)
        end = max(0, min(end, num_frames - 1))
        if end >= num_frames - 1:
            return num_frames, {}

        # 构造 frames_ranges
        times = entry["frame_idx"]
        bool_diff = (im_idx[:-1] != im_idx[1:])
        indices = bool_diff.nonzero().view(-1) + 1
        frames_ranges = torch.cat([
            torch.tensor([0], device=device),
            indices,
            torch.tensor([num_preds], device=device)
        ]).long()
        k = frames_ranges.size(0) - 1
        for i in range(k - 1, 0, -1):
            diff = int(im_idx[frames_ranges[i]] - im_idx[frames_ranges[i - 1]])
            if diff > 1:
                repeated = torch.tensor([frames_ranges[i]] * (diff - 1), device=device)
                frames_ranges = torch.cat((frames_ranges[:i], repeated, frames_ranges[i:]))
        if im_idx[0] > 0:
            repeated2 = torch.tensor([0] * int(im_idx[0].item()), device=device)
            frames_ranges = torch.cat((repeated2, frames_ranges))
        if frames_ranges.size(0) != num_frames + 1:
            needed = (num_frames + 1 - frames_ranges.size(0))
            repeated3 = torch.tensor([num_preds] * needed, device=device)
            frames_ranges = torch.cat((frames_ranges, repeated3))

        if end >= num_frames - 1 or frames_ranges[end] == frames_ranges[end + 1]:
            return num_frames, {}

        # Step 3: 提取 Scene 信息并分组
        subj_class, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices = self.dsgdetr.print_indices(entry)
        observed_anno = self.build_frames_annotation(im_idx, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices)[:end+1]
        # observed_anno = self.build_frames_annotation(im_idx, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices)[:end]
        future_anno = gt_annotation[end+1:]
        num_future = num_frames - end - 1
        num_objs = frames_ranges[end + 1] - frames_ranges[end]
        gt_annotation = gt_annotation[:end+1]

        obs_segments = self._merge_frames_for_objects_inference(observed_anno)
        obs_by_obj = self._group_segments_by_object_inference(obs_segments)

        # 获取最后一帧出现的对象
        end_frame_objects = []  # 改用list而不是set来保持顺序
        if len(observed_anno) > 0:
            last_frame = observed_anno[-1]
            for obj in last_frame[1:]:  # 从索引1开始遍历(跳过第0个frame meta信息)
                if 'class' in obj and 0 <= obj['class'] < len(self.obj_classes):
                    obj_class = self.obj_classes[obj['class']]
                    if obj_class not in end_frame_objects:  # 仍然需要去重，但保持首次出现的顺序
                        end_frame_objects.append(obj_class)
        all_objects = end_frame_objects

        # Step 4: 使用 Stage 1 预测所有对象的未来 Scene Graph
        prompts = []
        obj_list = []
        for obj_cls in all_objects:
            obs_obj_segments = obs_by_obj.get(obj_cls, [])
            if len(obs_obj_segments) == 0:
                continue
            observed_text = self._build_text_from_segments_for_object(obs_obj_segments, obj_cls, observed=True).split("\n")
            full_prompt = self.stage1_anticipator.build_prompt_for_scene_grap_old(
                observed_segments=observed_text,
                object_class=obj_cls,
                relationship_categories=self.relationship_categories,
                num_future_frames=num_future
            )
            prompts.append(full_prompt)
            obj_list.append(obj_cls)

        if len(prompts) == 0:
            return num_frames, {}

        max_attempts = 3  # 最大尝试次数
        # 用于存储每个 prompt 对应的生成结果（list of lines）
        results = [None] * len(prompts)
        # pending_indices 存储尚未成功生成结果的 prompt 索引
        pending_indices = list(range(len(prompts)))

        attempt = 0
        while attempt < max_attempts and pending_indices:
            # 针对当前 pending 的 prompts 一次性生成文本
            batch_prompts = [prompts[i] for i in pending_indices]
            batch_generated = self.stage1_anticipator.generate_text(
                prompts=batch_prompts,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.4
            )
            if isinstance(batch_generated, str):
                batch_generated = [batch_generated]
            
            # 保存本次未成功的索引（在本轮之后仍需要重试的）
            new_pending = []
            for idx_in_batch, gen_text in enumerate(batch_generated):
                i_obj = pending_indices[idx_in_batch]
                # 去除原始 prompt 部分
                gen_text = gen_text.replace(prompts[i_obj], "")
                lines_raw = self._split_generated_to_lines(gen_text)
                lines_parsed = self.extract_future_segments(lines_raw)
                needed = num_future
                if len(lines_parsed) >= needed:
                    lines_use = lines_parsed[:needed]
                else:
                    lines_use = lines_parsed[:]
                    while len(lines_use) < needed:
                        # 如果 lines_use 为空，则追加空字符串，否则复制最后一个
                        lines_use.append(lines_use[-1] if lines_use else "")
                # 检查是否有空字符串
                if any(line == '' for line in lines_use):
                    print(f"Prompt {i_obj} 尝试 {attempt + 1} 次后仍包含空字符串，准备在后续重试...")
                    new_pending.append(i_obj)
                else:
                    results[i_obj] = lines_use
            pending_indices = new_pending
            attempt += 1

        # 如果经过多次尝试仍有生成失败的 prompt，则返回空结果或其他处理
        if any(r is None for r in results):
            print("警告：经过 3 次尝试后，仍存在生成失败的结果。")
            return num_frames, {}

        # 将所有生成结果按原顺序拼接
        lines_batch = []
        obj_line_ranges = []
        running_idx = 0
        for res in results:
            start_idx = running_idx
            lines_batch.extend(res)
            running_idx += len(res)
            end_idx = running_idx
            obj_line_ranges.append((start_idx, end_idx))


        # Step 5: 使用 Stage 2 分配 Scene Graph
        future_frames = list(range(end+1, end+1+num_future))
        assigned_scene_graphs = []
        prompts = []
        object_details = []  # 用于存储每个对象相关的信息，便于后续解析

        # 1. 统一构建所有对象的 prompt
        for i_obj, obj_cls in enumerate(obj_list):
            obs_obj_segments = obs_by_obj.get(obj_cls, [])
            start_idx, end_idx = obj_line_ranges[i_obj]
            obj_lines = lines_batch[start_idx:end_idx]  # 提取该对象的 Scene Graph 行
            observed_text = self._build_text_from_segments_for_object(
                obs_obj_segments, obj_cls, observed=True, include_time=True
            ).split("\n")
            prompt = self.stage2_allocator.build_prompt(
                seg_text=observed_text,
                obj_lines=obj_lines,
                obj_cls=obj_cls,
                future_frames=future_frames
            )
            prompts.append(prompt)
            # 保存当前对象需要用于后续解析的信息（如 obj_lines）
            object_details.append((obj_cls, obj_lines))

        # 2. 一次性生成所有 prompt 对应的文本
        # 假设 generate_text 能接受 prompt 列表，并返回与之对应的生成文本列表
        generated_texts = self.stage2_allocator.generate_text(prompts, device=device)

        # 3. 对每个生成的文本进行解析，获得分配后的 Scene Graph
        for i_obj, gen_text in enumerate(generated_texts):
            _, obj_lines = object_details[i_obj]
            assigned_sg_for_obj = self.stage2_allocator.parse_generated_text(
                generated_text=gen_text,
                obj_lines=obj_lines,
                future_frames=future_frames
            )
            assigned_scene_graphs.extend(assigned_sg_for_obj)



        # Step 6: 处理分配后的 Scene Graph 并计算分布
        if self.use_classify_head:
            # dist_mat_all = self.classify_generated_text_for_object(lines_batch, None, device)  # [B, 26]
            dist_mat_all = self.classify_generated_text_for_object(assigned_scene_graphs, None, device=device)
        else:
            dist_mat_all = self.classify_generated_text_for_object_wo_classification_head(lines_batch, None)

        # ---------- (E) 拆分 => distribution_dict[obj_cls] => shape [num_future, 26] ----------
        distribution_dict = {}
        for i_obj, obj_cls in enumerate(obj_list):
            start_idx, end_idx = obj_line_ranges[i_obj]
            # 取出 [start_idx : end_idx] => shape [M, 26]
            # M <= num_future
            dist_mat_obj = dist_mat_all[start_idx:end_idx, :]
            # 还要把 shape 扩展到 [num_future, 26]，不足则最后行复制
            M = dist_mat_obj.size(0)
            if M < num_future:
                # 复制最后一行
                if M > 0:
                    last_row = dist_mat_obj[-1].unsqueeze(0)  # [1, 26]
                    repeat_count = num_future - M
                    pad_rows = last_row.repeat(repeat_count, 1)  # [repeat_count, 26]
                    dist_mat_obj = torch.cat([dist_mat_obj, pad_rows], dim=0)
                else:
                    # dist_mat_obj 是空 => 全0
                    dist_mat_obj = torch.zeros(num_future, dist_mat_all.size(1), device=dist_mat_all.device)
            elif M > num_future:
                # 理论上不会出现，因为我们只取了前 num_future 行
                dist_mat_obj = dist_mat_obj[:num_future, :]

            # frames => end+1 .. end+ num_future
            # 记下来
            frame_indices = list(range(end+1, end+1+num_future))
            distribution_dict[obj_cls] = {
                "dist":   dist_mat_obj,    # shape [num_future, 26]
                "frames": torch.tensor(frame_indices, device=dist_mat_obj.device)
            }
        # Step 7: 生成关系分布和最终预测
        pred = {}
        attn_mat, spat_mat, cont_mat = self.generate_relationship_distributions(distribution_dict, end, num_future)

        # return end+1, pred
        # 准备 build_pred_from_future_frames 的输入
        im_idx_list = [len(obj_list)] * num_future  # 每帧的对象数量等于 obj_list 长度
        labels_list = [[self.obj_classes.index(obj_cls) for obj_cls in obj_list]] * num_future  # 每帧的 object 标签

        # Step 8: 调用 build_pred_from_future_frames 构造 pred
        pred = self.build_pred_from_future_frames(
            num_future_frames=num_future,
            im_idx_list=im_idx_list,
            labels_list=labels_list,
            attn_mat=attn_mat,
            spat_mat=spat_mat,
            cont_mat=cont_mat,
            device=device
        )

        return end + 1, pred

        
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

    def generate_relationship_distributions(
        self,
        distribution_dict,
        end: int,
        num_future: int,
        obj_list: list = None,
    ):
        """
        根据给定的 distribution_dict（按对象存储未来分布）和若干辅助信息，
        生成三种关系分布( attention/spatial/contact )，并保证帧优先排列：
        frame0 => obj1, obj2, ... objN
        frame1 => obj1, obj2, ... objN
        ...
        frame(num_future-1) => ...
        
        同时构造 pair_idx, im_idx, 以及 boxes 占位。

        参数：
        distribution_dict: dict[obj_cls -> {"dist": Tensor[ K, 26 ], "frames": Tensor[K]}]
            - 这里存储每个对象在若干帧的关系分布(26维)，及对应帧编号 frames。
        end: 观测截止帧索引 (如 obs 段结束)
        num_future: 需要预测的未来帧数
        base_pair_idx: 基础的对象对索引 (shape=[num_objs, 2])，用来为每个对象复制
        num_objs: 对象数量 (frame优先中，每帧我们都排列 num_objs 行)
        obj_list: 对象列表(可选，若不传则从 distribution_dict.keys() 排序后使用)

        返回：
        attn_tensor, spat_tensor, cont_tensor: 分别形状 [num_future * num_objs, A / S / C]
        im_idx_tensor: shape [num_future * num_objs]
        pair_idx_tensor: shape [num_future * num_objs, 2]
        boxes_tensor: shape [max_index, 5] (占位)
        """
        
        # 如果未指定 obj_list，就对 distribution_dict 的 key 排序后用
        if obj_list is None:
            obj_list = distribution_dict.keys()

        A = self.attention_class_num
        S = self.spatial_class_num
        C = self.contact_class_num
        num_rel_classes = A + S + C  # 26

        attn_list = []
        spat_list = []
        cont_list = []

        # ----------------------------
        # 帧优先: 先 frame0 => obj1..objN, 再 frame1 => obj1..objN, ...
        # ----------------------------
        for i_frame in range(num_future):
            # 对每个对象(在同一帧)
            for j_obj, obj_cls in enumerate(obj_list):
                dist_info = distribution_dict[obj_cls]
                dist_mat  = dist_info["dist"]   # [K, 26]
                frames    = dist_info["frames"] # [K]

                # 目标帧： end+1 + i_frame (绝对帧号)
                target_frame_idx = (end+1) + i_frame

                # 从 dist_mat 中找到对应的行(若无则复制最后一行)
                row_26d = self._find_or_replicate(dist_mat, frames, target_frame_idx, num_rel_classes)

                # 拆分到 attn/spat/cont
                attn_vec = row_26d[:A]
                spat_vec = row_26d[A:A+S]
                cont_vec = row_26d[A+S:A+S+C]

                attn_list.append(attn_vec)
                spat_list.append(spat_vec)
                cont_list.append(cont_vec)

        # 组装 tensor
        attn_tensor = torch.stack(attn_list, dim=0)
        spat_tensor = torch.stack(spat_list, dim=0)
        cont_tensor = torch.stack(cont_list, dim=0)

        return attn_tensor, spat_tensor, cont_tensor

    def _find_or_replicate(self, dist_mat, frames, target_frame_idx, num_rel_classes):
        """
        在 dist_mat ([K, 26]) 中寻找 frames==target_frame_idx 的行；
        若找到则返回该行，否则复制 dist_mat 的最后一行(或返回全零).
        """
        # 确保 frames 是 PyTorch 张量
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, device=dist_mat.device)

        # frames shape=[K], dist_mat shape=[K, 26]
        mask = (frames == target_frame_idx)
        idxs = mask.nonzero(as_tuple=True)[0]  # 找到所有匹配的下标

        if len(idxs) > 0:
            # 若匹配到多行, 取第一行
            row_idx = idxs[0].item()
            return dist_mat[row_idx]
        else:
            # 若没有匹配 => 复制最后一行
            if dist_mat.size(0) > 0:
                return dist_mat[-1]
            else:
                # dist_mat 为空 => 返回零行
                return torch.zeros(num_rel_classes, dtype=torch.float32, device=dist_mat.device)

    def build_pair_idx_im_idx_and_boxes(
        self,
        pair_idx: torch.Tensor,
        frames_ranges: torch.Tensor,
        end: int,
        num_future: int,
        num_objs: int,
        device: torch.device
    ):
        """
        生成 pair_idx，确保未来帧索引从 0 递增，不错位。
        """
        # 1) 取出当前帧的 pair_idx
        slice_start = frames_ranges[end]
        slice_end   = frames_ranges[end + 1]
        pair_slice = pair_idx[slice_start:slice_end].clone()

        # 2) 归一化索引，使得 `pair_idx` 从 0 开始编号
        min_val = torch.min(pair_slice)
        pair_slice -= min_val  # 归一化索引，确保索引从 `0` 开始

        # 3) 复制 pair_slice 到所有未来帧
        repeated_slice = pair_slice.unsqueeze(0).repeat(num_future, 1, 1).view(-1, 2)

        # 4) 计算新的索引偏移量，确保未来帧索引从 `0` 递增
        offset_per_frame = torch.arange(num_future, device=device) * num_objs
        offset_per_frame = offset_per_frame.repeat_interleave(pair_slice.size(0)).unsqueeze(-1)

        # 5) 计算新的 `pair_idx`
        new_pair_idx = repeated_slice + offset_per_frame  # 确保从 0 递增

        # 6) 计算 im_idx，每帧索引递增
        im_idx_tensor = torch.arange(num_future, device=device).repeat_interleave(pair_slice.size(0))

        # 7) 计算 boxes
        max_index = int(new_pair_idx.max().item() + 1)
        boxes = torch.ones((max_index, 5), device=device) * 0.5

        return new_pair_idx, im_idx_tensor, boxes
    
    def extract_future_segments(self, lines):
        """
        从给定的 lines 列表中解析并提取符合以下格式的行：
        time [start..end]: <obj> XXX Attn=[...], Spat=[...], Cont=[...]

        并且只在“Subsequent frames:”之后的行里进行匹配。

        参数：
        lines (List[str]): 整段生成文本split("\n")得到的行列表。

        返回：
        future_lines (List[str]): 匹配到的未来帧行，每行为：
            time [3..5]: <obj> floor Attn=[looking_at], Spat=[behind], Cont=[holding]
        如果未找到 “Subsequent frames:” 或没有符合正则的行，则返回空列表。
        """

        # 1) 找到“Subsequent frames:”所在的行，确立未来帧起点
        
        start_index = 0


        pattern = _PATTERN_LINE
        future_lines = []
        for line in lines[start_index:]:
            line = line.strip()
            match = pattern.search(line)
            if match:
                # 提取匹配的子串（即 match.group(0)）
                future_lines.append(match.group(0))
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

    def _build_text_from_segments_for_object(self, obj_segments, obj_cls, observed=True, include_time=False):
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
            line = self._construct_segment_text(start_time, seg['end_time'], seg, obj_cls, include_time=include_time, add_obj_marker=True, ignore_obj_mode=False)
            lines.append(line)
        return "\n".join(lines)

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
        # text = f"{time_text}{obj_text} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
        text = f"{time_text}{obj_text} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
        # if not ignore_obj_mode:
        #     text = f"{time_text}{obj_text} Person attention to {obj_cls}: {attn_str}, {obj_cls} located relative to person: {spat_str}, Person contact with {obj_cls}: {cont_str}."
        # else:
        #     text = f"{time_text}Person attention to the object: {attn_str}, the object located relative to person: {spat_str}, Person contact with the object: {cont_str}."
        return text
        
    def _split_generated_to_lines(self, generated_text):
        lines = generated_text.split("\n")
        lines = [ln.strip() for ln in lines if ln.strip()]
        return lines

    def classify_generated_text_for_object_wo_classification_head(self, lines, obj_cls):
        """
        直接解析每一行文本以获取关系标签，返回一个二维Tensor [N, 26]，其中每行包含对应位置为1的标签。
        假设每行格式类似于：
        "Future segment 1, time from 503 to 503, Object[broom] Attention: not_looking_at, Spatial: in_front_of,on_the_side_of, Contact: holding."
        """
        num_classes = len(self.rel_classes)
        if not lines:
            return torch.empty((0, num_classes), dtype=torch.float32)

        result = []

        for line in lines:
            # 初始化长度为 num_classes 的零向量
            row = np.zeros(num_classes, dtype=np.float32)

            try:
                # 提取 Attention, Spatial, Contact 部分
                # 分割行以提取三种关系描述
                parts = line.split("Attention:")
                if len(parts) > 1:
                    after_attn = parts[1]
                    attn_split = after_attn.split("Spatial:")
                    attn_part = attn_split[0].strip().rstrip(',')

                    if len(attn_split) > 1:
                        after_spat = attn_split[1]
                        spat_split = after_spat.split("Contact:")
                        spat_part = spat_split[0].strip().rstrip(',')

                        if len(spat_split) > 1:
                            after_cont = spat_split[1]
                            cont_part = after_cont.strip().rstrip('.')
                        else:
                            cont_part = ""
                    else:
                        spat_part = ""
                        cont_part = ""
                else:
                    attn_part = ""
                    spat_part = ""
                    cont_part = ""

                # 将提取出的部分分割成单个关系
                attn_rels = [rel.strip() for rel in attn_part.split(',') if rel.strip()] if attn_part else []
                spat_rels = [rel.strip() for rel in spat_part.split(',') if rel.strip()] if spat_part else []
                cont_rels = [rel.strip() for rel in cont_part.split(',') if rel.strip()] if cont_part else []

                # 对每个关系查找索引并置1
                for rel in attn_rels:
                    if rel in self.rel_classes:
                        row[self.rel_classes.index(rel)] = 1
                for rel in spat_rels:
                    if rel in self.rel_classes:
                        row[self.rel_classes.index(rel)] = 1
                for rel in cont_rels:
                    if rel in self.rel_classes:
                        row[self.rel_classes.index(rel)] = 1

            except Exception as e:
                print(f"Error parsing line: {line}, error: {e}")
                # 如果解析失败，保持全零

            result.append(row)

        return torch.tensor(result, dtype=torch.float32)

    def classify_generated_text_for_object(self, lines, obj_cls, device=None):
        """
        对每一行文本进行 tokenize，然后采用平均池化整行隐藏状态，送入分类头获得 26 维关系预测，
        并通过 sigmoid 得到概率分布。返回的 shape 为 [N, 26]。
        
        参数：
        lines: List[str]，生成文本中按行拆分后的文本列表。
        obj_cls: 当前对象类别（这里不再使用 <obj> 标记，因此该参数可以仅作为辅助信息）。
        
        返回：
        一个 tensor，形状为 [N, NUM_REL_CLASSES]，包含每一行的预测概率。
        """
        if not lines:
            return torch.empty((0, self.attention_class_num + self.spatial_class_num + self.contact_class_num), 
                                dtype=torch.float32)
        
        # 对所有行文本 tokenize
        enc = self.stage1_anticipator.tokenizer(
            lines,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            # 这里调用模型的 forward，获取最后一层隐藏状态
            outputs = self.stage1_anticipator.joint_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                output_hidden_states=True,
                return_dict=True,
            )
        
        hidden_states = outputs.hidden_states[-1]  # shape: [B, seq_len, hidden_size]
        # 利用 attention mask 做平均池化
        attn_mask = enc["attention_mask"].unsqueeze(-1).float()  # shape: [B, seq_len, 1]
        pooled = (hidden_states * attn_mask).sum(dim=1) / (attn_mask.sum(dim=1) + 1e-9)  # shape: [B, hidden_size]
        
        with torch.no_grad():
            # 通过分类头计算 logits，再经过 sigmoid 得到概率
            logits = self.stage1_anticipator.joint_model.classifier(pooled)  # [B, NUM_REL_CLASSES]
            probs = torch.sigmoid(logits)
        
        return probs.cpu()

    def _clip_distribution_to_video(self, lines, dist_tensor, num_frames, end):
        """
        根据行顺序重新分配时间索引：lines[0]对应时间0，lines[1]对应时间1，……
        只保留时间索引 < num_frames 的行。
        返回 (filtered_dist, frame_indices)
        """
        filtered_list = []
        frame_indices = []

        # 计算可用的最大行数，不超过 num_frames
        available_frames = min(len(lines), num_frames)

        # 根据行顺序逐一分配时间索引，并收集对应的分布
        for i in range(available_frames):
            filtered_list.append(dist_tensor[i])
            frame_indices.append(end + i + 1)
        
        # 如果行数不足，复制最后一行
        if len(filtered_list) > 0 and len(filtered_list) < num_frames:
            last_dist = filtered_list[-1]
            while len(filtered_list) < num_frames:
                filtered_list.append(last_dist)
                frame_indices.append(end + len(filtered_list))

        # 如果没有任何有效行，返回空张量与空列表
        if not filtered_list:
            return torch.empty((0, dist_tensor.size(1)), dtype=dist_tensor.dtype), []

        # 将收集的分布列表堆叠成张量
        final_dist = torch.stack(filtered_list, dim=0)
        return final_dist, frame_indices
    
        # In SceneSayerODE class
    def convert_to_distribution(self, assigned_sg_for_obj):
        """
        Convert assigned Scene Graph texts to a multi-label distribution matrix.
        
        Parameters:
            assigned_sg_for_obj (list): List of assigned Scene Graph texts for each future frame.
        
        Returns:
            torch.Tensor: A tensor of shape [num_future, NUM_REL_CLASSES] representing the distribution.
        """
        dist_list = []
        for sg_text in assigned_sg_for_obj:
            label = np.zeros(NUM_REL_CLASSES, dtype=np.float32)
            # attn_match = re.search(r"Person attention to [^:]+: ([^,]+)", sg_text)
            # spat_match = re.search(r"located relative to person: ([^,]+)", sg_text)
            # cont_match = re.search(r"Person contact with [^:]+: ([^,\.]+)", sg_text)
            attn_match = re.search(r"attention:\s*([^,]+)", sg_text)
            spat_match = re.search(r"spatial:\s*([^,]+)", sg_text)
            cont_match = re.search(r"contact:\s*([^,\.]+)", sg_text)
            if attn_match:
                attn_str = attn_match.group(1)
                for attn in attn_str.split(','):
                    attn = attn.strip()
                    if attn in REL_CLASSES:
                        label[REL_CLASSES.index(attn)] = 1
            if spat_match:
                spat_str = spat_match.group(1)
                for spat in spat_str.split(','):
                    spat = spat.strip()
                    if spat in REL_CLASSES:
                        label[REL_CLASSES.index(spat)] = 1
            if cont_match:
                cont_str = cont_match.group(1)
                for cont in cont_str.split(','):
                    cont = cont.strip()
                    if cont in REL_CLASSES:
                        label[REL_CLASSES.index(cont)] = 1
            dist_list.append(label)
        dist_mat = np.stack(dist_list, axis=0)
        return torch.tensor(dist_mat, dtype=torch.float32)
    
    def build_pred_from_future_frames(self, num_future_frames, im_idx_list, labels_list,
                                    attn_mat, spat_mat, cont_mat, device):
        r"""
        根据未来帧的信息构造 scene graph 预测结果。

        输入参数：
            num_future_frames (int): 未来预测的帧数。
            im_idx_list (list[int]): 长度为 num_future_frames 的列表，每个元素为该帧的对象数量。
            labels_list (list[list[int]]): 长度为 num_future_frames 的列表，每个元素为该帧中所有对象的类别标签。
                                        注意：subject 的标签固定为 1，会自动添加在每一帧的首位，
                                        因此此处只需给出各帧中 object 的标签。
            attn_mat (torch.Tensor): 形状为 [total_objects, 3]，表示所有未来帧中对象的 attention 分布。
            spat_mat (torch.Tensor): 形状为 [total_objects, 6]，表示 spatial 分布。
            cont_mat (torch.Tensor): 形状为 [total_objects, 17]，表示 contacting 分布。
            device (torch.device): 指定生成 tensor 的设备。

        返回：
            pred (dict): 包含以下 key：
                - "labels": 1D tensor, 所有节点的标签（每帧第一个节点为 subject，固定为 1，其余为 object 标签）。
                - "scores": 1D tensor, 每个节点的分数（这里统一设为 1.0）。
                - "im_idx": 1D tensor, 长度等于所有未来帧中对象的总数，每个元素表示该对象所属的未来帧编号（从 0 开始）。
                - "pair_idx": 2D tensor, 每一行为 [subject_idx, object_idx]，表示每一帧中 subject 与各 object 的关联关系。
                - "boxes": 2D tensor, 占位符 boxes，形状为 [(num_future_frames + 总对象数), 5]，均初始化为 0.5。
                - "attention_distribution": 2D tensor, 形状为 [总对象数, 3]。
                - "spatial_distribution": 2D tensor, 形状为 [总对象数, 6]。
                - "contacting_distribution": 2D tensor, 形状为 [总对象数, 17]。
        """
        global_labels = []   # 存放各帧的节点标签（subject + objects）
        global_scores = []   # 存放各节点的分数
        global_pair_idx = [] # 存放每一帧中 subject 与各 object 的 pair (subject_idx, object_idx)
        global_im_idx = []   # 仅针对对象，记录每个 object 所属的未来帧编号

        current_idx = 0  # 全局节点索引计数器；每一帧节点数 = 1 (subject) + num_objs
        for f in range(num_future_frames):
            num_objs = im_idx_list[f]  # 当前帧中的 object 数量
            # 每一帧的节点顺序：第一个为 subject（固定标签 1），后续为各 object（标签由 labels_list 给出）
            frame_labels = [1] + labels_list[f]
            global_labels.extend(frame_labels)
            # 对应每个节点的分数，均设为 1.0
            frame_scores = [1.0] * (num_objs + 1)
            global_scores.extend(frame_scores)
            # 仅记录对象的所属帧编号（subject 不记录）
            global_im_idx.extend([f] * num_objs)
            # 构造当前帧中 subject 与每个 object 的 pair_idx
            for i in range(1, num_objs + 1):
                global_pair_idx.append([current_idx, current_idx + i])
            current_idx += (num_objs + 1)

        # 构造 boxes 占位符：行数 = 未来帧的 subject 数（num_future_frames） + 所有 object 数
        num_nodes = num_future_frames + sum(im_idx_list)
        boxes = torch.ones((num_nodes, 5), device=device) * 0.5

        pred = {}
        pred["labels"] = torch.tensor(global_labels, dtype=torch.long, device=device)
        pred["scores"] = torch.tensor(global_scores, dtype=torch.float32, device=device)
        pred["im_idx"] = torch.tensor(global_im_idx, dtype=torch.int32, device=device)
        pred["pair_idx"] = torch.tensor(global_pair_idx, dtype=torch.long, device=device)
        pred["boxes"] = boxes

        # 直接使用提供的 relationship distribution matrices
        total_objects = sum(im_idx_list)
        if total_objects > 0:
            pred["attention_distribution"] = attn_mat.to(device)    # shape: [total_objects, 3]
            pred["spatial_distribution"] = spat_mat.to(device)      # shape: [total_objects, 6]
            pred["contacting_distribution"] = cont_mat.to(device)   # shape: [total_objects, 17]
        else:
            pred["attention_distribution"] = torch.empty((0, 3), device=device)
            pred["spatial_distribution"] = torch.empty((0, 6), device=device)
            pred["contacting_distribution"] = torch.empty((0, 17), device=device)

        return pred