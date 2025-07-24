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
                 llama_path=None,
                 lora_path=None,
                 classifier_path=None,
                 use_fusion=False,
                 save_path=False):
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

        self.relationship_categories = {
            "Attention": ATTN_REL_CLASSES,
            "Spatial": SPAT_REL_CLASSES,
            "Contact": CONT_REL_CLASSES
            }
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
        print("SceneGraphAnticipator loaded.")

    def forward(self, entry, testing=False):
        """
        修改版：一次性生成 + 一次性分类
        - 对每个对象只调用一次 self.llm_anticipator.generate_text，让它输出 window 帧的关系描述。
        - 将解析到的各帧行一起调用 classify_generated_text_for_object，获得 [window, 26] 批量预测结果。
        - 最后再以 iwin=1..window 拼接进 entry。
        """
        device = entry["im_idx"].device

        # 1) 先用 DS-GDETR 处理当前帧
        entry = self.dsgdetr(entry)

        # 2) 获取基本信息
        im_idx        = entry["im_idx"]          # [N]
        pair_idx      = entry["pair_idx"]        # [N,2]
        gt_annotation = entry["gt_annotation"]   # list of frames
        num_frames    = len(gt_annotation)

        # 如果只有1帧，则无未来可预测
        if num_frames < 2:
            return entry

        # 3) 设置 window，若 self.max_window<=0 就默认 (num_frames-1)
        if self.max_window and self.max_window > 0:
            window = min(self.max_window, num_frames - 1)
        else:
            window = num_frames - 1

        # 从 DS-GDETR 中拿到 主体/客体类别 以及三种关系的 argmax
        subj_class, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices = \
            self.dsgdetr.print_indices(entry)

        # 4) 将 DS-GDETR 结果转成 frames_annotation 格式
        frames_anno_all = self.build_frames_annotation(
            im_idx, 
            obj_class,
            attn_rel_indices,
            sp_rel_indices,
            cont_rel_indices
        )
        # 分段 & 按对象分组
        all_segments = self._merge_frames_for_objects_inference(frames_anno_all)
        segs_by_obj  = self._group_segments_by_object_inference(all_segments)

        # 假设使用“最后一帧”出现的对象作为预测目标
        last_frame_data = frames_anno_all[-1]   
        last_objects = set()
        if len(last_frame_data)>1:
            for od in last_frame_data[1:]:
                cidx = od.get("class", -1)
                if 0 <= cidx < len(self.obj_classes):
                    last_objects.add(self.obj_classes[cidx])

        # -----------------------------------------------
        # 先对每个对象做 一次性 LLM 生成 & 一次性分类
        # 然后把 [window, 26] 结果存下来
        # -----------------------------------------------
        # dist_for_objects: dict[obj_cls -> Tensor[window, 26]]
        dist_for_objects = {}

        for obj_cls in last_objects:
            segs_for_obj = segs_by_obj.get(obj_cls, [])
            if not segs_for_obj:
                continue

            # (a) 构造观测段的文本提示
            observed_text = self._build_text_from_segments_for_object(
                segs_for_obj, obj_cls, observed=True
            ).split("\n")

            # (b) 用大模型一次性生成 window 帧描述
            prompt = self.llm_anticipator.build_prompt_for_scene_graph(
                observed_segments=observed_text,
                object_class=obj_cls,
                relationship_categories=self.relationship_categories,
                num_future_frames=window
            )
            # 为了让模型输出多帧，可以在 Prompt 中明确让它
            # "Predict the relationships for the next {window} frames"
            # 你可在 build_prompt_for_scene_graph 里或这里加上

            generated_text = self.llm_anticipator.generate_text(prompt, max_new_tokens=512)
            lines = self._split_generated_to_lines(generated_text)

            # (c) 解析出多帧 => 每帧若干行
            # 你可以用一个自定义函数一次性解析，如:
            # frames_dict[i] = list_of_lines_for_frame_i
            frames_dict = self._extract_all_future_frames(lines, window)

            # (d) 将 frames_dict 里全部帧的行合并成一个大列表 lines_batch，以便一次性分类
            #     同时记录每个帧在 lines_batch 里的起止下标 => 解析后拆分
            lines_batch = []
            frame_ranges = []  # list of (start_idx, end_idx)
            running = 0
            for iwin in range(1, window+1):
                lines_i = frames_dict.get(iwin, [])
                start_ = running
                end_   = running + len(lines_i)
                frame_ranges.append((start_, end_))
                running = end_
                lines_batch.extend(lines_i)

            if len(lines_batch) == 0:
                # 若完全没有文本 => 直接填零
                dist_for_objects[obj_cls] = torch.zeros(window, self.num_rel_classes, device=device)
                continue

            # (e) 一次性调用分类头 => [B, 26]， B = sum of lines in lines_batch
            if self.use_classify_head:
                dist_mat_all = self.classify_generated_text_for_object(lines_batch, obj_cls)  # [B, 26]
            else:
                dist_mat_all = self.classify_generated_text_for_object_wo_classification_head(lines_batch, obj_cls)

            # (f) 将 dist_mat_all 拆分到各帧 => 合并(或平均) => 得到 frame i => [26]
            #     这里示例只取第一行，也可做平均/最大
            per_frame_dist = []
            for iwin in range(1, window+1):
                start_, end_ = frame_ranges[iwin-1]
                if end_ > start_:
                    # 取第一行
                    row_26 = dist_mat_all[start_]
                else:
                    # 如果没有行 => 全零
                    row_26 = torch.zeros(self.num_rel_classes, device=device)
                per_frame_dist.append(row_26)
            # stack => [window, 26]
            dist_2d = torch.stack(per_frame_dist, dim=0)  
            dist_for_objects[obj_cls] = dist_2d  # [window, 26]

        # -----------------------------------------------
        # 6) 再用 "if iwin==1 => 赋值；else => cat" 逻辑，
        #    把 dist_for_objects 里的 [window, 26] 合并到 entry
        # -----------------------------------------------
        for iwin in range(1, window+1):
            # 对所有对象在这个帧 iwin 的 distribution 进行堆叠 => [Nobj, 26]
            # 然后拆分 attention/spatial/contact
            a_list, s_list, c_list = [], [], []
            for obj_cls in dist_for_objects:
                row_26 = dist_for_objects[obj_cls][iwin-1]  # shape [26]
                a_vec = row_26[: self.attention_class_num]
                s_vec = row_26[self.attention_class_num : self.attention_class_num + self.spatial_class_num]
                c_vec = row_26[self.attention_class_num + self.spatial_class_num : ]
                a_list.append(a_vec)
                s_list.append(s_vec)
                c_list.append(c_vec)

            if len(a_list) > 0:
                attn_dist = torch.stack(a_list, dim=0)
                spat_dist = torch.stack(s_list, dim=0)
                cont_dist = torch.stack(c_list, dim=0)
            else:
                attn_dist = torch.empty((0, self.attention_class_num), device=device)
                spat_dist = torch.empty((0, self.spatial_class_num), device=device)
                cont_dist = torch.empty((0, self.contact_class_num), device=device)

            # “if iwin==1 => 赋值；else => cat”
            if iwin == 1:
                entry["anticipated_attention_distribution"]   = attn_dist
                entry["anticipated_spatial_distribution"]     = spat_dist
                entry["anticipated_contacting_distribution"]  = cont_dist
            else:
                entry["anticipated_attention_distribution"] = torch.cat([
                    entry["anticipated_attention_distribution"], attn_dist
                ], dim=0)
                entry["anticipated_spatial_distribution"] = torch.cat([
                    entry["anticipated_spatial_distribution"], spat_dist
                ], dim=0)
                entry["anticipated_contacting_distribution"] = torch.cat([
                    entry["anticipated_contacting_distribution"], cont_dist
                ], dim=0)

        # 7) 返回 entry，其中包含 anticipated_xxx_distribution
        return entry
    # 推理时主入口
    def forward_single_entry(self, context_fraction, entry):
        """
        推理主入口 (合并所有对象的 prompt 一次性推理):
        1. 利用 dsgdetr 处理当前帧，获取 im_idx、pair_idx、gt_annotation、frame_idx 等信息。
        2. 根据 context_fraction 计算观测端 end 帧，分割出未来段 future_anno 并计算 num_future。
        3. 解析观测段、得到最后一帧出现的对象 all_objects；
            对 each object => 构造 prompt => (一次性)合并到 prompts 列表。
        4. 调用 self.llm_anticipator.generate_text(prompts=..., ...) => 得到 generated_texts 列表 (batch_size=len(all_objects))。
        5. 对 generated_texts 中的每一条文本 => split => extract_future_segments => 只取前 num_future 行 => 收集到 lines_batch，并记录 obj_line_ranges[i] = (start_idx, end_idx)。
        6. 若 self.use_classify_head=True => classify_generated_text_for_object(lines_batch) => 返回 dist_mat_all [B, 26]；否则走无分类头解析。
        7. 将 dist_mat_all 拆分回各对象 => [num_future, 26] => distribution_dict[obj_cls] = {...}
        8. 调用 generate_relationship_distributions(...) + build_pair_idx_im_idx_and_boxes(...) => pred
        9. 返回 (end+1, pred)
        """
        device = entry["im_idx"].device

        # 1) 先跑 DS-GDETR 处理
        entry = self.dsgdetr(entry)

        im_idx        = entry["im_idx"]
        pair_idx      = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds     = im_idx.size(0)
        num_frames    = len(gt_annotation)

        if num_frames < 2:
            # 无法预测未来
            return num_frames, {}

        # 2) 根据 context_fraction => end
        end = int(torch.ceil(torch.tensor(num_frames * context_fraction)).item() - 1)
        end = max(0, min(end, num_frames - 1))
        if end >= num_frames - 1:
            return num_frames, {}  # 没有未来帧

        # 构造 frames_ranges (平展), 这是你原有的逻辑
        times       = entry["frame_idx"]
        bool_diff   = (im_idx[:-1] != im_idx[1:])
        indices     = bool_diff.nonzero().view(-1) + 1
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

        # 1) 提取 scene info
        subj_class, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices = self.dsgdetr.print_indices(entry)
        observed_anno = self.build_frames_annotation(
            im_idx, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices
        )[: end+1]

        future_anno   = gt_annotation[end+1:]  # [end+1 .. last]
        num_future    = num_frames - end - 1
        num_objs      = frames_ranges[end + 1] - frames_ranges[end]
        gt_annotation = gt_annotation[: end+1]

        save_dir = "SceneSayer/results_llama_3b/fusion"
        has_changes = self.record_object_changes(
            observed_anno=observed_anno,
            future_anno=future_anno,
            obj_classes=self.obj_classes,
            save_dir=save_dir
            )

        # 2.5) 分别合并 & 按对象分组
        obs_segments = self._merge_frames_for_objects_inference(observed_anno)
        obs_gt_segments = self._merge_frames_for_objects_inference(gt_annotation)
        fut_segments = self._merge_frames_for_objects_inference(future_anno)
        obs_by_obj   = self._group_segments_by_object_inference(obs_segments)
        obs_gt_by_obj = self._group_segments_by_object_inference(obs_gt_segments)
        fut_by_obj   = self._group_segments_by_object_inference(fut_segments)

        # 获取最后一帧出现的 objects
        end_frame_objects = []
        end_gt_frame_objects = []
        if len(observed_anno) > 0:
            last_frame = observed_anno[-1]
            last_gt_frame = gt_annotation[-1]
            for obj in last_frame[1:]:
                if 'class' in obj:
                    cidx = obj['class']
                    if 0 <= cidx < len(self.obj_classes):
                        end_frame_objects.append(self.obj_classes[cidx])
            for obj in last_gt_frame[1:]:
                if 'class' in obj:
                    cidx = obj['class']
                    if 0 <= cidx < len(self.obj_classes):
                        end_gt_frame_objects.append(self.obj_classes[cidx])

        # 只考虑最后一帧中出现的 objects
        all_objects = end_frame_objects
        all_gt_objects = end_gt_frame_objects

        #obs_gt_by_obj中的对象替换掉obs_by_obj的对应项
        for obj_cls in all_gt_objects:
            obs_by_obj[obj_cls] = copy.deepcopy(obs_gt_by_obj[obj_cls])

        # # ---------- (A) 合并 prompts ----------
        prompts = []
        obj_list = []
        for obj_cls in all_objects:
            obs_obj_segments = obs_by_obj.get(obj_cls, [])
            if len(obs_obj_segments) == 0:
                continue

            # 构造 prompt
            # 使用未来的
            observed_text = self._build_text_from_segments_for_object(obs_obj_segments, obj_cls, observed=True).split("\n")
            full_prompt = self.llm_anticipator.build_prompt_for_scene_graph(
                observed_segments=observed_text,
                object_class=obj_cls,
                relationship_categories=self.relationship_categories,
                num_future_frames=num_future
            )
            prompts.append(full_prompt)
            obj_list.append(obj_cls)
        
        # --------------debug----------------
        future_text = []
        fut_obj_list = []
        for obj_cls in all_gt_objects:
            fut_gt_obj_segments = fut_by_obj.get(obj_cls, [])
            if len(fut_gt_obj_segments) == 0:
                continue
            future_text.append(self._build_text_from_segments_for_object(fut_gt_obj_segments, obj_cls, observed=True).split("\n"))
            fut_obj_list.append(obj_cls)
        # --------------end--------------

        if len(prompts) == 0:
            # 说明没有对象 => 无预测
            return end+1, {}

        # ---------- (B) 一次性 generate => list[str], len= len(obj_list) ----------
        generated_texts = self.llm_anticipator.generate_text(
            prompts=prompts,
            max_new_tokens=1024,   # 或其他合适值
            temperature=0.7,
            top_p=0.95
        )
        # 如果只输入一个 prompt，会返回 str，否则 list[str]
        # 这里肯定是 list[str]，因为 prompts 不止一个
        if isinstance(generated_texts, str):
            generated_texts = [generated_texts]

        # ---------- (C) 解析 => lines_batch, 记录行范围 ----------
        lines_batch = []
        obj_line_ranges = []  # [(start_idx, end_idx), ...] same length as obj_list
        running_idx = 0

        # 逐对象解析
        for i_obj, gen_txt in enumerate(generated_texts):
            # --------------- original --------------------
            # gen_txt = gen_txt.replace(prompts[i_obj],"")
            # lines_raw = self._split_generated_to_lines(gen_txt)
            # # 提取可用行
            # lines_parsed = self.extract_future_segments(lines_raw)
            # ---------------- end ----------------------
            # ---------------- debug ---------------------
            if obj_list[i_obj] in fut_obj_list:
                lines_raw = future_text[fut_obj_list.index(obj_list[i_obj])]
                lines_parsed = self.extract_future_segments(lines_raw)
            else:
                gen_txt = gen_txt.replace(prompts[i_obj],"")
                lines_raw = self._split_generated_to_lines(gen_txt)
                # 提取可用行
                lines_parsed = self.extract_future_segments(lines_raw)
            # ---------------- end ----------------------

            # user要保证“截取需要预测帧数 n 行”，若 lines_parsed 太多，只取前 n 行
            # 如果不足 n 行，后面可以在 cat 时做对应
            needed = num_future
            if len(lines_parsed) >= needed:
                # 只取前 n 行
                lines_use = lines_parsed[:needed]
            else:
                lines_use = lines_parsed  # 可能不够
                # 补全 lines_use 到 needed 长度
                while len(lines_use) < needed:
                    if lines_use:
                        lines_use.append(lines_use[-1])  # 复制最后一行
                    else:
                        lines_use.append("")  # 添加空字符串


            start_idx = running_idx
            lines_batch.extend(lines_use)
            running_idx += len(lines_use)
            end_idx   = running_idx

            obj_line_ranges.append((start_idx, end_idx))

        # 现在 lines_batch 的总行数 = sum of min(num_future, lines_parsed_for_each_obj)
        B = len(lines_batch)  # 期望 B = sum_i lines_use[i].size
        # ---------- debug ----------
        # 根据fut_segments直接生成相应的文本，替换掉由llm生成的相应object的文本
        observed_text = self._build_text_from_segments_for_object(fut_segments, obj_cls, observed=True).split("\n")
        # ---------- (D) 一次性行分类: dist_mat_all => [B, 26] ----------
        if self.use_classify_head:
            dist_mat_all = self.classify_generated_text_for_object(lines_batch, None)  # [B, 26]
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

        # ---------- (F) 组装 => generate_relationship_distributions => pred ----------
        pred = {}
        attn_mat, spat_mat, cont_mat = self.generate_relationship_distributions(
            distribution_dict,
            end=end,
            num_future=num_future
        )
        new_pair_idx, im_idx_tensor, boxes = self.build_pair_idx_im_idx_and_boxes(
            pair_idx=pair_idx,
            frames_ranges=frames_ranges,
            end=end,
            num_future=num_future,
            num_objs=num_objs,
            device=device
        )

        if attn_mat.shape[0] == 0:
            pred["attention_distribution"] = torch.empty((0, self.attention_class_num), device=device)
            pred["spatial_distribution"]   = torch.empty((0, self.spatial_class_num), device=device)
            pred["contacting_distribution"]= torch.empty((0, self.contact_class_num), device=device)
            pred["im_idx"]   = torch.empty((0,), dtype=torch.int32, device=device)
            pred["pair_idx"] = torch.empty((0, 2), dtype=torch.int64, device=device)
            pred["boxes"]    = torch.empty((0, 5), device=device)
        else:
            pred["attention_distribution"]  = attn_mat.to(device)
            pred["spatial_distribution"]    = spat_mat.to(device)
            pred["contacting_distribution"] = cont_mat.to(device)
            pred["im_idx"]   = torch.tensor([i for i in range(num_frames - end - 1) for j in range(frames_ranges[end + 1] - frames_ranges[end])], dtype=torch.int32).to(device=frames_ranges.device)
            mx = torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) - torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1
            pred["pair_idx"] = (pair_idx[frames_ranges[end] : frames_ranges[end + 1]] - torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]])).repeat(num_frames - end - 1, 1) + mx * torch.reshape(pred["im_idx"], (-1, 1))
            pred["boxes"] = torch.ones(mx * (num_frames - end - 1), 5).to(device=im_idx.device) / 2

        # 如果需要复制 scores/labels:
        min_idx = torch.min(pair_idx[frames_ranges[end]: frames_ranges[end+1]])
        max_idx = torch.max(pair_idx[frames_ranges[end]: frames_ranges[end+1]]) + 1
        repeated_count = num_future
        if self.mode == "predcls":
            pred["scores"] = entry["scores"][min_idx:max_idx].repeat(repeated_count)
            pred["labels"] = entry["labels"][min_idx:max_idx].repeat(repeated_count)
        else:
            pred["pred_scores"] = entry["pred_scores"][min_idx:max_idx].repeat(repeated_count)
            pred["pred_labels"] = entry["pred_labels"][min_idx:max_idx].repeat(repeated_count)

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

        # 2) 定义匹配新 prompt 行格式的正则：
        #    例如：time [3..5]: <obj> floor Attn=[looking_at], Spat=[on_the_side_of], Cont=[being_held]
        pattern = re.compile(
            r"object:\s*(\w+)\s*Attention:\s*(\w+),\s*Spatial:\s*(\w+),\s*Contact:\s*(\w+)",
            flags=re.IGNORECASE
        )
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
        enc = self.llm_anticipator.tokenizer(
            lines,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.llm_anticipator.device)
        
        with torch.no_grad():
            # 这里调用模型的 forward，获取最后一层隐藏状态
            outputs, _ = self.llm_anticipator.joint_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                output_hidden_states=True,
                return_dict=True,
                do_classification=False
            )
        
        hidden_states = outputs.hidden_states[-1]  # shape: [B, seq_len, hidden_size]
        # 利用 attention mask 做平均池化
        attn_mask = enc["attention_mask"].unsqueeze(-1).float()  # shape: [B, seq_len, 1]
        pooled = (hidden_states * attn_mask).sum(dim=1) / (attn_mask.sum(dim=1) + 1e-9)  # shape: [B, hidden_size]
        
        with torch.no_grad():
            # 通过分类头计算 logits，再经过 sigmoid 得到概率
            logits = self.llm_anticipator.joint_model.classifier(pooled)  # [B, NUM_REL_CLASSES]
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
    
    def record_object_changes(self, observed_anno, future_anno, obj_classes, save_dir):
        """
        记录观测帧的最后一帧与未来预测帧的对象变化。
        
        Args:
            observed_anno (list): 观测帧的标注数据
            future_anno (list): 未来预测帧的标注数据  
            obj_classes (list): 对象类别列表
            save_dir (str): CSV文件保存路径
            
        Returns:
            bool: 如果有对象变化则返回True,否则返回False
        """
        import os
        import pandas as pd
        
        # 1. 获取最后观测帧的对象ID
        last_frame_obj_ids = []
        if len(observed_anno) > 0:
            last_frame = observed_anno[-1]
            for obj in last_frame[1:]:
                if 'class' in obj:
                    cidx = obj['class']
                    if 0 <= cidx < len(obj_classes):
                        last_frame_obj_ids.append(str(cidx))
        last_frame_obj_set = set(last_frame_obj_ids)
        last_frame_obj_str = ";".join(sorted(last_frame_obj_ids))

        # 2. 获取每个预测帧的对象并检查新对象
        future_frame_objects = []
        has_new_objects = False
        for frame in future_anno:
            frame_objs = []
            for obj in frame[1:]:
                if 'class' in obj:
                    cidx = obj['class']
                    if 0 <= cidx < len(obj_classes):
                        frame_objs.append(str(cidx))
                        # 检查是否出现新物体
                        if str(cidx) not in last_frame_obj_set:
                            has_new_objects = True
            if frame_objs:
                future_frame_objects.append(";".join(sorted(frame_objs)))

        # 3. 检查是否所有未来帧都与最后观测帧相同
        all_same = True
        for future_objs in future_frame_objects:
            if future_objs != last_frame_obj_str:
                all_same = False
                break

        # 4. 如果对象有变化，则记录到CSV
        if not all_same:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, "object_changes.csv")
            
            # 准备新行数据，has_new_objects放在第一位
            new_row = {
                'has_new_objects': 1 if has_new_objects else 0,
                'last_observed_objects': last_frame_obj_str
            }
            for i, future_objs in enumerate(future_frame_objects):
                new_row[f'future_frame_{i+1}_objects'] = future_objs
                
            # 读取现有CSV或创建新的
            try:
                df = pd.read_csv(csv_path)
                new_idx = df['idx'].max() + 1 if len(df) > 0 else 1
            except FileNotFoundError:
                df = pd.DataFrame()
                new_idx = 1
                
            new_row['idx'] = new_idx
            
            # 确保列的顺序：先 idx，然后 has_new_objects，再是其他列
            if len(df) == 0:
                # 如果是新文件，设置列的顺序
                columns = ['idx', 'has_new_objects', 'last_observed_objects'] + \
                        [f'future_frame_{i+1}_objects' for i in range(len(future_frame_objects))]
                df = pd.DataFrame(columns=columns)
                
            # 添加新行并保存
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # 确保保存时维持列的顺序
            if 'idx' in df.columns:
                column_order = ['idx', 'has_new_objects', 'last_observed_objects'] + \
                            [col for col in df.columns if col.startswith('future_frame_')]
                df = df[column_order]
            df.to_csv(csv_path, index=False)
            
            return True
            
        return False
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
        self.tokenizer.padding_side = 'left'

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


    def build_prompt_for_scene_graph(self, observed_segments, object_class, relationship_categories, few_shot_example=None, num_future_frames=1):
            n = 8
            example_obs = None
            example_fut = None
            if few_shot_example is None:
                if len(observed_segments) > 2*n:
                    example_obs = "\n".join(observed_segments[:n])
                    
                elif len(observed_segments) > n:
                    n = n // 2
                    example_obs = "\n".join(observed_segments[:n])

                if len(observed_segments) > (n+num_future_frames):
                    example_fut = "\n".join(observed_segments[n:n+num_future_frames])
                
                if example_obs is not None and example_fut is not None:
                    few_shot_example = (
                        "Example:\n"
                        f"Observed {n} segments for object [{object_class}]:\n"
                        f"{example_obs}\n"
                        f"Future {num_future_frames} segments for object [{object_class}]:\n"
                        f"{example_fut}\n"
                    )
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
            # if few_shot_example:
            #     header += few_shot_example + "\n"
            instruction_0 = (
                f"\nPlease generate the future segment for object [{object_class}] "
                "in the same structured format as above. "
                "Do not add extra commentary; output exactly in the given style.\n"
            )
            observed_text = f"Observed segment for object [{object_class}]:\n" + "\n".join(observed_segments) + "\n"
            instruction = f"Future {num_future_frames} segments for object [{object_class}]:\n"
            prompt = header + instruction_0 + observed_text + instruction
            # prompt = instruction_0 + observed_text + instruction
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
        decoded = [self.tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(outputs.size(0))]
        return decoded[0] if single_input else decoded
