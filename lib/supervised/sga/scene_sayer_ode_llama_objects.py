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
                #  llama_path="SceneSayer/llama/Llama-3.2-3B-Instruct",
                 llama_path="SceneSayer/llama/DeepSeek-R1-Distill-Qwen-1.5B",
                 lora_path="SceneSayer/llama_SGA/results/deepseek_1.5b/0.9_prompt/epoch_10",
                 classifier_path="SceneSayer/llama_SGA/results/deepseek_1.5b/0.9_prompt/epoch_10/classifier.bin"):
        """
        新增: llama_path, lora_path 用于初始化 SceneGraphAnticipator
        """
        super(SceneSayerODE, self).__init__()
        self.mode = mode

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attn_rel_classes = ATTN_REL_CLASSES
        self.spat_rel_classes = SPAT_REL_CLASSES
        self.cont_rel_classes = CONT_REL_CLASSES
        self.attention_class_num = attention_class_num
        self.spatial_class_num   = spatial_class_num
        self.contact_class_num   = contact_class_num
        self.use_classify_head = use_classify_head
        # 旧逻辑: self.diff_func = get_derivatives(...) 已不需要

        self.d_model = 1936
        if script_required and object_required:
            self.d_model += 768
        elif script_required:
            self.d_model += 256

        self.max_window = max_window

        # 1) 保留 STTran，用于处理当前帧
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

        # 2) 用 JointModel + 分类头，而不是简单 PeftModel
        #    => 通过 SceneGraphAnticipator 进行加载
        self.llm_anticipator = SceneGraphAnticipator(
            model_path=llama_path,
            lora_path=lora_path,
            classifier_path=classifier_path,
            device="cuda",
            FP16=False
        )


    def build_known_frames_text(self, entry, step_i: int):
        """
        将当前帧的场景图信息转换为自然语言描述，
        并在上下文中包含历史预测结果（如果存在）。
        同时显式告诉 LLM：有哪些对象，需要保留。
        """
        device = entry["im_idx"].device

        # 1) 从 entry 中获取 (subj_class, obj_class, rel_indices)
        subj_class, obj_class, attn_rel_indices, spaitial_rel_indices, contacting_rel_indices = self.dsgdetr.print_indices(entry)

        # 根据 pair_idx => 统计出现过的 object label
        # 也可直接取 subj_class, obj_class 并去重
        all_obj_indices = torch.cat([subj_class, obj_class], dim=0)
        unique_obj_labels = []
        for idx_obj in all_obj_indices.cpu().tolist():
            if idx_obj < len(self.obj_classes):
                lab = self.obj_classes[idx_obj]
                if lab not in unique_obj_labels:
                    unique_obj_labels.append(lab)

        # 2) 获取场景图标签信息
        scene_info = self.dsgdetr.get_scene_graph_labels(obj_class, attn_rel_indices, spaitial_rel_indices, contacting_rel_indices)

        lines = []
        count_objs = len(scene_info["objects"])
        for i in range(count_objs):
            obj_label = scene_info["objects"][i]
            attn_rel_label = scene_info["attn_relationships"][i]
            spatial_rel_label = scene_info["spatial_relationships"][i]
            contact_rel_label = scene_info["contacting_relationships"][i]
            time = int(entry["im_idx"][i])

            # 构建当前帧的描述
            line = (f"time t{time}: Object[{obj_label}] "
                    f"Attention: {attn_rel_label}, "
                    f"Spatial: {spatial_rel_label}, "
                    f"Contact: {contact_rel_label}")
            lines.append(line)

        # 将当前帧描述拼接
        current_frame_text = " || ".join(lines)

        # 3) 显式提示“不要增删对象”
        #    （也可加入：可用的关系、可用的 time 规则等等）
        num_unique_objs = len(unique_obj_labels)
        obj_list_str = ", ".join(unique_obj_labels)
        header_line = (
            f"We have {num_unique_objs} object(s): {obj_list_str}. "
            "Please do NOT add or remove objects in subsequent frames.\n"
        )

        known_text = current_frame_text

        # 4) 若存在历史预测 => 将其加到 prompt
        if 'predicted_history' in entry and entry['predicted_history']:
            history_lines = []
            for hist in entry['predicted_history']:
                time_length = len(hist.get('time', []))
                object_length = len(hist.get('object_class', []))
                attention_length = len(hist.get('attention_rels', []))
                spatial_length = len(hist.get('spatial_rels', []))
                contact_length = len(hist.get('contact_rels', []))

                for i in range(time_length):
                    time_val = hist['time'][i] if i < time_length else "unknown"
                    obj_label = "unknown"
                    if i < object_length and hist['object_class'][i] is not None:
                        obj_label = hist['object_class'][i]

                    # 这里 hist['attention_rels'][i] 是一个list[str], 可能是 []
                    # 先判空
                    if i < attention_length and hist['attention_rels'][i]:
                        attn_rels = ", ".join(hist['attention_rels'][i])
                    else:
                        attn_rels = "None"

                    if i < spatial_length and hist['spatial_rels'][i]:
                        spat_rels = ", ".join(hist['spatial_rels'][i])
                    else:
                        spat_rels = "None"

                    if i < contact_length and hist['contact_rels'][i]:
                        cont_rels = ", ".join(hist['contact_rels'][i])
                    else:
                        cont_rels = "None"

                    hist_line = (f"time t{time_val}: Object[{obj_label}] "
                                f"Attention Relationship: {attn_rels}, "
                                f"Spatial Relationship: {spat_rels}, "
                                f"Contact Relationship: {cont_rels}")
                    history_lines.append(hist_line)

            # 将历史预测信息拼接并添加到已知文本中
            history_text = " || ".join(history_lines)
            known_text = known_text + " || " + history_text

        return known_text, header_line

    def parse_future_struct_and_fill(self, future_struct, device, n=1):
        """
        核心修改点：不再做 0/1 填充，而是对每“行”文本做分类头forward -> 得到分布 -> 切分成 attn/spat/cont
        """
        # 先保持你原先的 time/obj/records/去重/补全逻辑 => final_records
        # ========== 原先 Steps A~C & E 中的一系列操作都可保留，以维护 time/padding 逻辑 ==========
        # 这里示例性地保留主要逻辑

        time_list = future_struct['time']
        obj_list  = future_struct['object_class']
        attn_list = future_struct['attention_rels']
        spat_list = future_struct['spatial_rels']
        cont_list = future_struct['contact_rels']

        records = []
        for i in range(len(time_list)):
            t_str = time_list[i]
            if t_str.startswith('t'):
                t_str = t_str[1:]
            try:
                t_val = int(t_str)
            except:
                t_val = -999
            records.append({
                'time': t_val,
                'obj_class': obj_list[i] if i<len(obj_list) else "unknown",
                'attn_rels': attn_list[i] if i<len(attn_list) else [],
                'spat_rels': spat_list[i] if i<len(spat_list) else [],
                'cont_rels': cont_list[i] if i<len(cont_list) else [],
            })

        # (可选)做 last_seen_dict 补全
        # (可选)做 distinct_times 限制
        # 这里省略具体实现，只保留 final_records

        final_records = records  # 简化：假设就全要

        # 构造 filtered_struct
        final_time  = []
        final_obj   = []
        final_attn  = []
        final_spat  = []
        final_cont  = []

        for r in final_records:
            final_time.append(f"t{r['time']}")
            final_obj.append(r['obj_class'])
            final_attn.append(r['attn_rels'])
            final_spat.append(r['spat_rels'])
            final_cont.append(r['cont_rels'])

        filtered_struct = {
            'time': final_time,
            'object_class': final_obj,
            'attention_rels': final_attn,
            'spatial_rels': final_spat,
            'contact_rels': final_cont
        }

        N = len(final_records)

        # =========== 核心：改用 “分类头” 的 logits => 分布 ===========

        # 先为每条 record 拼出一个文本 => 输入到分类模型
        # 你可以用 parse 出来的 "time t{...}: Object[...] Attention Relationship:...,..." 之类
        # 下面仅做简易示例
        record_texts = []
        for r in final_records:
            # 可以把 attn_rels/spat_rels/cont_rels 拼到一行 => 让分类头自己判断
            # 也可以只把 obj_class/time 拼进去
            # 这里示例：把整行信息合并成字符串
            line_text = f"time t{r['time']}, object[{r['obj_class']}], attn={r['attn_rels']}, spat={r['spat_rels']}, cont={r['cont_rels']}"
            record_texts.append(line_text)

        # 用 llm_anticipator.classify_text(...) 一次性批量获取 [N, NUM_REL_CLASSES]
        big_probs = self.llm_anticipator.classify_text(record_texts)  # shape=[N, 25], device=cpu

        # 分割成 attn / spat / cont
        attn_dist = torch.zeros(N, self.attention_class_num, device=device)
        spat_dist = torch.zeros(N, self.spatial_class_num, device=device)
        cont_dist = torch.zeros(N, self.contact_class_num, device=device)

        # 这里假设: 
        # - attention_class_num=3 => 对应 REL_CLASSES 索引 [0,1,2]
        # - spatial_class_num=5  => 对应 REL_CLASSES 索引 [3..7]
        # - contact_class_num=17 => 对应 REL_CLASSES 索引 [8..24]
        # 你可根据自己实际来
        for i in range(N):
            # big_probs[i] => [25]
            row = big_probs[i].to(device)  # => [25] on GPU
            attn_dist[i] = row[0 : self.attention_class_num]  # 0..2
            spat_dist[i] = row[self.attention_class_num : (self.attention_class_num+self.spatial_class_num)]
            cont_dist[i] = row[(self.attention_class_num+self.spatial_class_num) : (self.attention_class_num+self.spatial_class_num+self.contact_class_num)]

        return attn_dist, spat_dist, cont_dist, filtered_struct

    def forward(self, entry, testing=False):
        """
        替换原先的 ODE 逻辑 => LLM 预测
        """
        device = entry["im_idx"].device  # 方便后面
        # 1) 先获取当前帧 scene graph
        entry = self.dsgdetr(entry)

        # 原先: obj = entry["pair_idx"][:,1], labeling, etc.
        if not testing:
            labels_obj = entry["labels"][ entry["pair_idx"][:,1] ]
        else:
            pred_labels_obj = entry["pred_labels"][ entry["pair_idx"][:,1] ]
            labels_obj      = entry["labels"][ entry["pair_idx"][:,1] ]

        im_idx        = entry["im_idx"]
        pair_idx      = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds     = im_idx.size(0)
        times         = entry["frame_idx"]  # already a Tensor
        # 2) 维持原 frames_ranges, mask, etc.
        #    只改 "预测"部分
        # ...
        bool_diff = (im_idx[:-1] != im_idx[1:])
        indices   = bool_diff.nonzero().view(-1)+1

        curr_id      = 0
        times_unique = torch.unique(torch.tensor(times)).float().to(device)
        num_frames   = len(gt_annotation)
        window       = self.max_window if self.max_window!=-1 else (num_frames-1)
        window       = min(window, num_frames-1)

        times_extend  = torch.arange(times_unique[-1]+1, times_unique[-1]+window+1, device=device)
        global_output = entry["global_output"]

        # 原先 anticipated_vals shape (window, 0, d_model)
        anticipated_vals = torch.zeros(window, 0, self.d_model, device=device)

        frames_ranges = torch.cat((
            torch.tensor([0], device=device),
            indices,
            torch.tensor([num_preds], device=device)
        ), dim=0).long()

        k = frames_ranges.size(0)-1
        for i in range(k-1, 0, -1):
            diff = int(im_idx[ frames_ranges[i]] - im_idx[ frames_ranges[i-1]])
            if diff>1:
                repeated = torch.tensor([frames_ranges[i]]*(diff-1), device=device)
                frames_ranges = torch.cat((frames_ranges[:i], repeated, frames_ranges[i:]))

        if im_idx[0]>0:
            repeated2 = torch.tensor([0]*int(im_idx[0].item()), device=device)
            frames_ranges = torch.cat((repeated2, frames_ranges))

        if frames_ranges.size(0) != num_frames+1:
            needed = (num_frames+1 - frames_ranges.size(0))
            repeated3 = torch.tensor([num_preds]*needed, device=device)
            frames_ranges = torch.cat((frames_ranges, repeated3))

        repeated_times = []
        for rr_i in range(frames_ranges.size(0)-1):
            seg_len = frames_ranges[rr_i+1] - frames_ranges[rr_i]
            if seg_len<0: seg_len=0
            repeated_times.append( times_unique[rr_i].unsqueeze(0).repeat(seg_len) )
        if len(repeated_times)>0:
            entry["times"] = torch.cat(repeated_times).to(device)
        else:
            entry["times"] = torch.empty(0, device=device)

        entry["rng"] = frames_ranges
        times_unique = torch.cat((times_unique, times_extend), dim=0)

        # 3) for i in range(1, window+1):
        for iwin in range(1, window+1):
            # mask
            mask_preds = torch.empty(0, dtype=torch.long, device=device)
            mask_gt    = torch.empty(0, dtype=torch.long, device=device)
            gt         = gt_annotation.copy()

            for j in range(num_frames - iwin):
                if testing:
                    a = pred_labels_obj[ frames_ranges[j]:frames_ranges[j+1] ]
                    b = labels_obj[ frames_ranges[j+iwin]:frames_ranges[j+iwin+1] ]
                else:
                    a = labels_obj[ frames_ranges[j]:frames_ranges[j+1] ]
                    b = labels_obj[ frames_ranges[j+iwin]:frames_ranges[j+iwin+1] ]

                # 做 intersection
                # a, b都是1D Tensor
                # approach: a_list, b_list
                a_list = a.cpu().tolist()
                b_list = b.cpu().tolist()
                intersect_vals = []
                # 先 build a dict for b
                freq_b = {}
                for val in b_list:
                    freq_b[val] = freq_b.get(val, 0)+1

                # 遍历 a_list
                matched_idx_a = []
                matched_idx_b = []
                for idx_a, val_a in enumerate(a_list):
                    if freq_b.get(val_a,0)>0:
                        matched_idx_a.append(idx_a)
                        freq_b[val_a] -=1
                # matched_idx_a => intersection indices in a
                # for b we do not store exact pos => might be a difference from original approach
                # 这里仅demo

                # convert to Torch
                offset_a = frames_ranges[j]
                matched_idx_a_t = torch.tensor(matched_idx_a, dtype=torch.long, device=device) + offset_a
                # matched_idx_b we skip for eq approach
                offset_b = frames_ranges[j+iwin]

                mask_preds = torch.cat([mask_preds, matched_idx_a_t])
                # we just do the same size for b?
                # 这里简单 assume same length
                # or do a second pass for b? 
                # in original code, we tried to match index in b
                # let's do simpler approach => we do the same length
                mask_gt_batch = torch.arange(len(matched_idx_a), device=device) + offset_b
                mask_gt = torch.cat([mask_gt, mask_gt_batch])

            entry[f"mask_curr_{iwin}"] = mask_preds
            entry[f"mask_gt_{iwin}"]   = mask_gt

            # if testing => fill last_{iwin}, etc. skip for brevity

            # ========== LLM 核心: 构造 Prompt => 预测 => parse => fill distribution
            known_text, head_line = self.build_known_frames_text(entry, step_i=iwin)

            future_text = self.llm_anticipator.anticipate_future_frames(
                known_frames_text=known_text,
                num_future_frames=1,

            )
            # parse
            future_structs = self.llm_anticipator.parse_generated_text_to_graph(future_text)
            # breakpoint()
            # 取第一条
            if len(future_structs)>0:
                fstruct = future_structs[0]
            else:
                fstruct = {
                    'time': [],
                    'object_class': [],
                    'attention_rels': [],
                    'spatial_rels': [],
                    'contact_rels': []
                }
            # 映射 => fill
            attn_dist, spat_dist, cont_dist, filtered_struct = self.parse_future_struct_and_fill(fstruct, device=device)
            
            # 赋给entry => "anticipated_*_distribution"
            # 注意 dimension: 
            # 这里仅示例 => 你可以合并
            # if iwin==1 => entry["anticipated_attention_distribution"] = ...
            # or do a stack => depends on baseline's usage
            # 这里演示: 直接写entry["anticipated_attention_distribution"] = attn_dist
            if iwin == 1:
                entry["anticipated_attention_distribution"]   = attn_dist
                entry["anticipated_spatial_distribution"]     = spat_dist
                entry["anticipated_contacting_distribution"]  = cont_dist
            else:
                # 叠加?
                # 也可 concat => depends on baseline
                entry["anticipated_attention_distribution"] = torch.cat([
                    entry["anticipated_attention_distribution"], attn_dist
                ], dim=0)
                entry["anticipated_spatial_distribution"] = torch.cat([
                    entry["anticipated_spatial_distribution"], spat_dist
                ], dim=0)
                entry["anticipated_contacting_distribution"] = torch.cat([
                    entry["anticipated_contacting_distribution"], cont_dist
                ], dim=0)

        return entry

        #############################
    # 推理时主入口
    def forward_single_entry(self, context_fraction, entry):
        """
        1) 用 dsgdetr 提取当前帧信息
        2) 按观测/未来帧 => 分割 annotation
        3) 对每个对象 => 构造 prompt => LLM 生成 => 行解析 => classify -> distribution
        4) 组装与“第二个版本 forward_single_entry”一致的 pred 字典:
            - attention_distribution, spatial_distribution, contacting_distribution
            - im_idx, pair_idx, boxes, ...
        5) return end+1, pred
        """
        device = entry["im_idx"].device

        # 1) 先跑 dsgdetr
        entry = self.dsgdetr(entry)

        im_idx        = entry["im_idx"]
        pair_idx      = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds     = im_idx.size(0)
        num_frames    = len(gt_annotation)

        if num_frames < 2:
            return num_frames, {}  # 无法预测

        # 2) 根据 context_fraction => end
        end = int(torch.ceil(torch.tensor(num_frames * context_fraction)).item() - 1)
        end = max(0, min(end, num_frames - 1))
        # 若 end 已到达最后，没未来帧可预测 => 提前返回
        if end >= num_frames - 1:
            return num_frames, {}
        

        times       = entry["frame_idx"]  # 1D
        bool_diff   = (im_idx[:-1] != im_idx[1:])
        indices     = bool_diff.nonzero().view(-1) + 1
        frames_ranges = torch.cat([
            torch.tensor([0], device=device),
            indices,
            torch.tensor([num_preds], device=device)
        ]).long()

        # 修正 frames_ranges (平展)
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

        # 1）提取scene information
        subj_class, obj_class, attn_rel_indices, spaitial_rel_indices, contacting_rel_indices = self.dsgdetr.print_indices(entry)
        observed_anno = self.build_frames_annotation(im_idx, obj_class, attn_rel_indices, spaitial_rel_indices, contacting_rel_indices)[:end+1]
        # 2）按 end 分割 annotation
        # observed_anno = gt_annotation[:end+1]
        future_anno   = gt_annotation[end+1:]  # [end+1 .. num_frames-1]
        num_future    = num_frames - end - 1
        num_objs   = frames_ranges[end + 1] - frames_ranges[end]  

        # 2.5) 分别合并 & 按对象分组
        obs_segments = self._merge_frames_for_objects_inference(observed_anno)
        fut_segments = self._merge_frames_for_objects_inference(future_anno)
        obs_by_obj = self._group_segments_by_object_inference(obs_segments)
        fut_by_obj = self._group_segments_by_object_inference(fut_segments)

        # all_objects = set(obs_by_obj.keys()) | set(fut_by_obj.keys())
        # 获取最后一帧的objects
        end_frame_objects = set()
        last_frame = observed_anno[-1]
        for obj in last_frame[1:]:
            if 'class' in obj:
                end_frame_objects.add(self.obj_classes[obj['class']])

        # 只考虑最后一帧中出现的objects
        all_objects = end_frame_objects

        
        distribution_dict = {}


        # 3) 逐对象 => 构造 prompt => LLM => classify => dist
        for obj_cls in all_objects:
            obs_obj_segments = obs_by_obj.get(obj_cls, [])
            fut_obj_segments = fut_by_obj.get(obj_cls, [])

            if len(obs_obj_segments) == 0:
                continue
            
            # if len(fut_obj_segments) == 0:
            #     # 无未来帧 => 跳过
            #     continue

            # (a) 构建 prompt
            prompt_text = self._build_text_for_inference(obs_obj_segments, obj_cls, observed=True)
            full_prompt =  (
                "Below are known frames: \n"
                f"{prompt_text}\n\n"
                "Now, please predict only the future frames in a concise format. \n"
                "Do NOT include extra analysis or reasons. \n"
                "Subsequent frames: \n"
            )

            # (b) 推理 -> 生成文本
            input_len = len(self.llm_anticipator.tokenizer(full_prompt)["input_ids"])

            max_retries = 3
            current_input_len = input_len
            retry_count = 0

            while retry_count < max_retries:
                generated_text = self.llm_anticipator.generate_text(
                    prompts=full_prompt,
                    max_new_tokens=(current_input_len + 512),
                    temperature=0.8,
                    top_p=0.95
                )

                init_lines = self._split_generated_to_lines(generated_text)
                lines = self.extract_future_segments(init_lines)

                if self.use_classify_head:
                    dist_tensor = self.classify_generated_text_for_object(lines, obj_cls)
                else:
                    dist_tensor = self.classify_generated_text_for_object_wo_classification_head(lines, obj_cls)

                final_dist, frame_indices = self._clip_distribution_to_video(lines, dist_tensor, num_future, end)
                
                if len(frame_indices) > 0:
                    break
                
                current_input_len += 200
                retry_count += 1

            if retry_count >= max_retries:
                # 达到最大重试次数，返回空结果
                final_dist = torch.empty((0, dist_tensor.size(1)), dtype=dist_tensor.dtype)
                frame_indices = []

            distribution_dict[obj_cls] = {
                "dist": final_dist,
                "frames": frame_indices
            }

        # -------------------------
        # 4) 组装成 pred 与“第二个版本”结构一致
        # -------------------------
        pred = {}
        attn_list = []
        # 遍历所有对象 => 拼到 pred
        # for obj_cls, dist_info in distribution_dict.items():
        #     dist = dist_info["dist"]         # shape [K, 26]
        #     frames = dist_info["frames"]     # len=K
        #     obj_idx = self.obj_classes.index(obj_cls)
        #     if dist.size(0) == 0:
        #         current_object_idx += 1
        #         continue

        #     for i in range(dist.size(0)):
        #         frame_idx = frames[i]  # 原始在 [0..num_frames-1]
        #         # 只处理“未来帧” => frame_idx >= end+1
        #         if frame_idx < (end+1):
        #             continue
        #         future_offset = frame_idx - (end + 1)  # => [0..(num_future-1)]
        #         if future_offset < 0 or future_offset >= num_future:
        #             # 超出范围 => 跳过
        #             continue

        #         # im_idx => future_offset
        #         im_idx_list.append(future_offset)

        #         # pair_idx => 当前对象索引 (如果您需要2D可以再扩展)
        #         pair_idx_list.append([current_object_idx, current_object_idx])

        #     current_object_idx += 1

        # 4) 逐对象组装
        attn_mat, spat_mat, cont_mat = self.generate_relationship_distributions(distribution_dict, end, num_future)

        new_pair_idx, im_idx_tensor, boxes = self.build_pair_idx_im_idx_and_boxes(
                                                                                pair_idx = pair_idx,
                                                                                frames_ranges = frames_ranges,
                                                                                end = end,
                                                                                num_future = num_future,
                                                                                num_objs = num_objs,
                                                                                device = device)

        # 组合
        if attn_mat.shape[0] == 0:
            # 说明 LLM 全部没预测到
            # 也可以直接 return
            pred["attention_distribution"] = torch.empty((0, self.attention_class_num), device=device)
            pred["spatial_distribution"]   = torch.empty((0, self.spatial_class_num), device=device)
            pred["contacting_distribution"]= torch.empty((0, self.contact_class_num), device=device)
            pred["im_idx"] = torch.empty((0,), dtype=torch.int32, device=device)
            pred["pair_idx"] = torch.empty((0, 2), dtype=torch.int64, device=device)
            pred["boxes"]   = torch.empty((0, 5), device=device)
        else:
            pred["attention_distribution"] = copy.deepcopy(attn_mat).to(device)
            pred["spatial_distribution"]   = copy.deepcopy(spat_mat).to(device)
            pred["contacting_distribution"]= copy.deepcopy(cont_mat).to(device)
            pred["pair_idx"] = copy.deepcopy(new_pair_idx).to(device)
            pred["boxes"]    = copy.deepcopy(boxes).to(device)
            pred["im_idx"]   = copy.deepcopy(im_idx_tensor).to(device)
            if pred["im_idx"].shape[0] != pred["attention_distribution"].shape[0]:
                print("im_idx and attention_distribution shape mismatch")


        # 如果还有 scores/labels 需要扩展:
        # 您可以参考“第二个版本”在 predcls 模式下如何复制 [scores,labels] 并填充
        # 这里只是演示:
        min_idx = torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]])
        max_idx = torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1
        repeated_count = num_future
        if self.mode == "predcls":
            pred["scores"] = entry["scores"][min_idx : max_idx].repeat(repeated_count)
            pred["labels"] = entry["labels"][min_idx : max_idx].repeat(repeated_count)
        else:
            pred["pred_scores"] = entry["pred_scores"][min_idx : max_idx].repeat(repeated_count)
            pred["pred_labels"] = entry["pred_labels"][min_idx : max_idx].repeat(repeated_count)
        

        return end+1, pred

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
            obj_list = sorted(distribution_dict.keys())

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
        复现“第二个版本”中对 pair_idx 的构造逻辑，并创建 boxes 占位，同时返回 im_idx。
        不依赖 enumerate(obj_list)，而是直接使用给定的 pair_idx + frames_ranges 计算。

        参数:
            pair_idx: 原始 DS-GDETR 的对象对索引 (形如 [N, 2])，与 frames_ranges 相匹配
            frames_ranges: 第 i 帧在 pair_idx 中的起始/结束区间 (shape [num_frames+1])
            end: 指定观测截止帧索引
            num_future: 要预测的未来帧数量
            num_objs: 第 end 帧内对象(或关系)数, frames_ranges[end+1] - frames_ranges[end]
            device: 计算设备

        返回:
            new_pair_idx: 形如 [num_future * num_objs, 2] 的张量
            im_idx_tensor: 形如 [num_future * num_objs] 的张量，对应未来帧编号
            boxes: 形如 [max_index, 5] 的张量 (占位框)
        """

        # 1) 取出第 `end` 帧的对象对索引
        slice_start = frames_ranges[end]
        slice_end   = frames_ranges[end + 1]
        pair_slice = pair_idx[slice_start : slice_end]

        # 2) 减去最小值 => 归一化
        min_val = torch.min(pair_slice)
        pair_slice = pair_slice - min_val  # 保证索引从 0 开始

        # 3) 将当前帧的对象对复制到所有未来帧
        repeated_slice = pair_slice.unsqueeze(0).repeat(num_future, 1, 1).view(-1, 2)
        # 形状变化: [M,2] => [1,M,2] => [num_future,M,2] => reshape => [num_future*M,2]
        # 其中 M = slice_end - slice_start = num_objs

        # 4) 计算 offset_im_idx 和 im_idx_tensor
        #    依次为 0,0,...,0(共 num_objs 次), 1,1,...,1(共 num_objs 次), ...
        offset_im_idx = torch.arange(num_future, device=device).view(num_future, 1)
        offset_im_idx = offset_im_idx.repeat(1, num_objs).view(-1)  # => [num_future * num_objs]
        im_idx_tensor = offset_im_idx.clone()  # 保存为 im_idx

        # 5) 将 offset 乘以 mx(= num_objs)，再加到 repeated_slice
        mx = num_objs
        repeated_slice_offset = repeated_slice + mx * offset_im_idx.unsqueeze(-1)
        # => shape [num_future * num_objs, 2]
        new_pair_idx = repeated_slice_offset

        # 6) 计算 boxes 的大小
        max_index = int(new_pair_idx.max().item() + 1)
        boxes = torch.ones((max_index, 5), device=device) * 0.5

        return new_pair_idx, im_idx_tensor, boxes
    
    def extract_future_segments(self, lines):
        """
        从给定的 lines 列表中解析并提取完整的未来帧预测行。
        要求格式：
        Future segment <数字>, time from <起始时间> to <结束时间>, Object[<对象名>] Attention: <...>, Spatial: <...>, Contact: <...>.
        """
        # 找到“Subsequent frames:”所在的索引，以确定未来帧部分的起点
        start_index = None
        for i, line in enumerate(lines):
            if "Subsequent frames:" in line:
                start_index = i + 1  # 未来帧信息从下一行开始
                break
        # 如果未找到相关标志，则返回空列表
        if start_index is None:
            return []

        # 定义匹配未来帧预测行的正则表达式模式
        pattern = re.compile(
            r'^Future segment \d+, time from \d+ to \d+, '
            r'Object\[[^\]]+\] Attention: .+?, Spatial: .+?, Contact: .+?\.$'
        )

        # 从找到的起点开始，筛选符合格式的行
        future_lines = []
        for line in lines[start_index:]:
            # 使用正则表达式匹配完整的未来帧预测行
            if pattern.match(line):
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

    def _build_text_for_inference(self, obj_segments, obj_cls, observed=True):
        if len(obj_segments)==0:
            return f"No known segments for object[{obj_cls}]"
        prefix = "Observed" if observed else "Future"
        lines = []
        end_time = None
        for i, seg in enumerate(obj_segments):
            attn_str = ",".join(str(self.rel_classes[a]) for a in seg["attn_ids"]) or "None"
            spat_str = ",".join(str(self.rel_classes[s]) for s in seg["spat_ids"]) or "None"
            cont_str = ",".join(str(self.rel_classes[c]) for c in seg["cont_ids"]) or "None"

            if i == 0:
                line = (
                    f"{prefix} segment {i+1}, time from {seg['start_time']} to {seg['end_time']}, "
                    f"Object[{obj_cls}] "
                    f"Attention: {attn_str}, Spatial: {spat_str}, Contact: {cont_str}."
                )
            else:
                line = (
                    f"{prefix} segment {i+1}, time from {end_time + 1} to {seg['end_time']}, "
                    f"Object[{obj_cls}] "
                    f"Attention: {attn_str}, Spatial: {spat_str}, Contact: {cont_str}."
                )
            end_time = seg["end_time"]
            lines.append(line)
        return "\n".join(lines)

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

    def classify_generated_text_for_object(self, lines, obj_cls):
        """
        逐行tokenize -> 找 <obj> -> gather hidden state -> 过分类头 -> 26维 => sigmoid
        返回 [N, 26]
        """
        if not lines:
            return torch.empty((0, self.attention_class_num+self.spatial_class_num+self.contact_class_num), dtype=torch.float32)

        enc = self.llm_anticipator.tokenizer(lines, padding=True, truncation=True, return_tensors='pt').to(self.llm_anticipator.device)
        with torch.no_grad():
            # 这里要看您 LLM 对接的 forward 是否返回 (lm_outputs, hidden_states) 或别的
            # 假设是 (lm_outputs, hidden_states)
            lm_outputs, _ = self.llm_anticipator.joint_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                output_hidden_states=True,
                return_dict=True,
                do_classification=False
            )
            hidden_states = lm_outputs.hidden_states[-1]  # [B, seq_len, hidden_size]

        obj_token_id = self.llm_anticipator.tokenizer.convert_tokens_to_ids("<obj>")
        batch_size = enc["input_ids"].size(0)
        embeddings = torch.zeros((batch_size, hidden_states.size(-1)), device=hidden_states.device)

        for i_line in range(batch_size):
            line_ids = enc["input_ids"][i_line]
            obj_pos = -1
            for t_i, token_id in enumerate(line_ids):
                if token_id.item() == obj_token_id:
                    obj_pos = t_i
                    break
            if obj_pos == -1:
                obj_pos = len(line_ids)-1
            embeddings[i_line] = hidden_states[i_line, obj_pos, :]

        with torch.no_grad():
            logits = self.llm_anticipator.joint_model.classifier(embeddings)  # => [B, 26]
            probs  = torch.sigmoid(logits)

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
#####################################
# 从“代码1”里复制过来的 JointModel 类
#####################################
class JointModel(nn.Module):
    """
    与代码1相同：包含 base_model(LoRA) + classifier。
    用于同时进行文本生成 & 分类头输出。
    """
    def __init__(self, base_model: nn.Module, classifier: nn.Module, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        return_dict=False,
        do_classification=False,
        classifier_positions="last_token",
    ):
        # 1) 跑 base_model (CausalLM)
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
            hidden_states = lm_outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_size]

            if classifier_positions == "last_token":
                pooled_emb = hidden_states[:, -1, :]  # [batch, hidden_size]
            else:
                # 也可做 average_pool
                pooled_emb = hidden_states.mean(dim=1)

            clf_logits = self.classifier(pooled_emb)  # => [batch, NUM_REL_CLASSES]

        return lm_outputs, clf_logits

#####################################
# SceneGraphAnticipator：加载 JointModel 而不是直接 PeftModel
#####################################
class SceneGraphAnticipator:
    """
    负责用大模型(带LoRA+分类头)进行文本生成 & 分类。
    """
    def __init__(self, model_path, lora_path, classifier_path, device='cuda', FP16=False):
        """
        Args:
          model_path: 例如 "/path/to/llama-3B"
          lora_path:  例如 "/path/to/lora_finetuned_ddp/epoch_2"
          classifier_path: "/path/to/lora_finetuned_ddp/epoch_2/classifier.bin"
        """
        self.device = device

        # 1) 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = 'left'
        self.tokenizer.padding_side = 'right'

        # 添加特殊标记
        special_tokens = {"additional_special_tokens": ["<obj>"]}
        self.tokenizer.add_special_tokens(special_tokens)

        # 2) 加载 base CausalLM
        if not FP16:
            base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        base_model.resize_token_embeddings(len(self.tokenizer))
        # 3) 加载 LoRA
        peft_model = PeftModel.from_pretrained(base_model, lora_path).to(device)
        # 设置不训练
        for p in peft_model.parameters():
            p.requires_grad_(False)
        peft_model.eval()

        # 4) 构建分类头 + JointModel
        hidden_size = peft_model.model.config.hidden_size
        classifier = nn.Linear(hidden_size, NUM_REL_CLASSES).to(device)
        # 加载分类头权重
        state_dict = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(state_dict)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)

        self.joint_model = JointModel(peft_model, classifier, hidden_size).eval().to(device)


    def build_prompt_for_scene_graph(self, observed_segments, object_class, relationship_categories, few_shot_example=None):
        """
        构建结构化的 prompt，用于 scene graph anticipation 任务（单 object）。

        参数：
          - observed_segments (List[str]): 针对该 object 的观测片段（每个字符串描述一个时间段的信息），例如：
                ["time [0..2]: <obj> cup Attn=[none], Spat=[on_table], Cont=[none]",
                 "time [3..4]: <obj> cup Attn=[glancing], Spat=[on_table], Cont=[none]"]
          - object_class (str): 对象名称，例如 "cup"、"bag" 等。
          - relationship_categories (dict): 各类别关系字典，例如：
                {
                    "Attention": ["looking_at", "not_looking_at", "unsure"],
                    "Spatial": ["in_front_of", "behind", "on_the_side_of"],
                    "Contact": ["holding", "touching", "none"]
                }
          - few_shot_example (str, optional): Few-shot 示例文本；若为 None，则使用默认示例。

        返回：
          - prompt (str): 构造好的结构化 prompt 字符串。

        说明：
          该 prompt 包括任务说明、关系类别列表、few-shot 示例、观测段及生成指令。
        """
        if few_shot_example is None:
            few_shot_example = (
                "Example:\n"
                "Observed segment for object [cup]:\n"
                "time [0..2]: <obj> cup Attn=[none], Spat=[on_table], Cont=[none]\n"
                "Future segment for object [cup]:\n"
                "time [3..5]: <obj> cup Attn=[looking_at], Spat=[in_front_of], Cont=[being_held]\n"
            )
        header = (
            "You are a scene graph anticipation assistant. All relationships are defined as the interaction "
            "between a person and an object.\n"
            "For the given object, you need to predict how the person will relate to it in the future.\n"
            "The possible relationship categories are:\n"
            f"  Attention: {', '.join(relationship_categories.get('Attention', []))}\n"
            f"  Spatial: {', '.join(relationship_categories.get('Spatial', []))}\n"
            f"  Contact: {', '.join(relationship_categories.get('Contact', []))}\n"
            "Follow the format exactly as shown in the example. Do NOT add extra commentary or explanation.\n\n"
        )
        header += few_shot_example + "\n"
        observed_text = f"Observed segment for object [{object_class}]:\n" + "\n".join(observed_segments) + "\n"
        instruction = (
            f"\nPlease generate the future segment for object [{object_class}] in the same structured format as above. "
            "Do not add extra commentary; output exactly in the given style.\n"
            "Subsequent frames:\n"
        )
        prompt = header + observed_text + instruction
        return prompt
    
    def generate_text(self, prompts, max_new_tokens=256, temperature=0.9, top_p=0.7):
        """
        用 self.joint_model.base_model 做文本生成。
        其中 base_model 是 peft_model, 也就是 LLaMA + LoRA
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            single_input = True
        else:
            single_input = False

        enc = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        input_len = enc["input_ids"].size(1)

        # 直接调用 self.joint_model.base_model.generate
        # self.joint_model.base_model 相当于 peft_model
        # TODO: 重新定义generate函数，使其支持隐藏状态输出
        outputs = self.joint_model.base_model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # decode
        decoded = []
        for i in range(outputs.size(0)):
            txt = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            decoded.append(txt)

        if single_input:
            return decoded[0]
        else:
            return decoded

    def classify_text(self, text_list):
        """
        对输入的每个文本做一次 forward(do_classification=True)，获得 [batch, 25] logits。
        返回 logits 或者 sigmoid(prob)都可以，这里返回sigmoid(prob)。
        """
        if isinstance(text_list, str):
            text_list = [text_list]
            single_input = True
        else:
            single_input = False

        enc = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            lm_outputs, clf_logits = self.joint_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                output_hidden_states=True,
                return_dict=True,
                do_classification=True
            )
            probs = torch.sigmoid(clf_logits)  # shape [batch_size, NUM_REL_CLASSES]

        probs = probs.detach().cpu()
        if single_input:
            return probs[0]
        else:
            return probs

    def anticipate_future_frames(
        self,
        known_frames_text,
        start_time: int = None,
        end_time: int = None,
        num_future_frames: int = None,
        length: int = 256,
        head_line: str = None
    ):
        """
        生成场景图描述的文本（与原本类似）。
        """
        # 同你原先写法
        if isinstance(known_frames_text, str):
            known_frames_text = [known_frames_text]
            single_input = True
        else:
            single_input = False

        prompts = []
        for text in known_frames_text:
            if start_time is not None and end_time is not None:
                prompt = (
                    "Below are the descriptions of known frames. "
                    f"Please write the scene graph descriptions for frames from time t{start_time} to time t{end_time}:\n\n"
                    f"{text}\n\n"
                    "Subsequent frame descriptions:"
                )
            elif num_future_frames is not None and head_line is not None:
                prompt = (
                    "Below are the descriptions of known frames. "
                    f"Please write the scene graph descriptions for the subsequent {num_future_frames} frames:\n\n"
                    f"{text}\n\n"
                    "Subsequent frame descriptions:"
                )
            else:
                prompt = (
                    "Below are the descriptions of known frames. "
                    "Please write the scene graph descriptions for the subsequent frames:\n\n"
                    f"{text}\n\n"
                    "Subsequent frame descriptions:"
                )
            prompts.append(prompt)

        # 调用 self.generate_text
        batch_future_texts = self.generate_text(
            prompts,
            max_new_tokens=length,
            temperature=0.9,
            top_p=0.7
        )

        if single_input:
            return batch_future_texts[0]
        else:
            return batch_future_texts

    def parse_generated_text_to_graph(self, generated_text):
        """
        与原本类似：正则解析 time / object / attn / spat / cont
        """
        if isinstance(generated_text, str):
            generated_text = [generated_text]

        time_pattern = r"time\s+t(\d+)"
        obj_pattern = (
            r"Object\[(.*?)\].*?"
            r"Attention Relationship:\s*(.*?)(?:,|$).*?"
            r"Spatial Relationship:\s*(.*?)(?:,|$).*?"
            r"(?:Contact Relationship|Contacting Relationship):\s*(.*?)(?:,|$)"
        )

        all_output_dicts = []

        for text_item in generated_text:
            output_dict = {
                'time': [],
                'object_class': [],
                'attention_rels': [],
                'spatial_rels': [],
                'contact_rels': []
            }

            frames = text_item.split("||")
            for frame in frames:
                frame = frame.strip()
                if not frame:
                    continue

                time_match = re.search(time_pattern, frame, flags=re.I)
                if not time_match:
                    continue
                time_num = time_match.group(1)
                output_dict['time'].append(f"t{time_num}")

                obj_matches = re.findall(obj_pattern, frame, flags=re.I)
                for obj_cls, attn_str, spat_str, cont_str in obj_matches:
                    output_dict['object_class'].append(obj_cls.strip())

                    def split_rels(s):
                        s = s.strip()
                        if s.lower()=='none':
                            return []
                        return [x.strip() for x in s.split(',') if x.strip()]

                    attn_rels = split_rels(attn_str)
                    spat_rels = split_rels(spat_str)
                    cont_rels = split_rels(cont_str)

                    output_dict['attention_rels'].append(attn_rels)
                    output_dict['spatial_rels'].append(spat_rels)
                    output_dict['contact_rels'].append(cont_rels)

            all_output_dicts.append(output_dict)
        return all_output_dicts
