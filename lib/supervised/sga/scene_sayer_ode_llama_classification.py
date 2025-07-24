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

# 你已有的 STTran 实现
from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierTransformer, GetBoxes
from lib.word_vectors import obj_edge_vectors
import time
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
                 llama_path="SceneSayer/llama/Llama-3.2-3B-Instruct",
                 lora_path="SceneSayer/llama_SGA/results/de_repeat/epoch_2",
                 classifier_path="SceneSayer/llama_SGA/results/de_repeat/epoch_2/classifier.bin"):
        """
        新增: llama_path, lora_path 用于初始化 SceneGraphAnticipator
        """
        super(SceneSayerODE, self).__init__()
        self.mode = mode

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num   = spatial_class_num
        self.contact_class_num   = contact_class_num

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
                    f"Attention Relationship: {attn_rel_label}, "
                    f"Spatial Relationship: {spatial_rel_label}, "
                    f"Contact Relationship: {contact_rel_label}")
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
    
    def get_n_list(self, total, chunk_size):
        """
        根据总帧数和每次希望预测的帧数，构建每次循环时的 n 列表。
        """
        lst = []
        remaining = total
        while remaining > 0:
            current = min(chunk_size, remaining)
            lst.append(current)
            remaining -= current
        return lst

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

    def forward_single_entry(self, context_fraction, entry):
        import time  # 确保在函数内可以访问time模块
        device = entry["im_idx"].device
        total_start = time.time()

        # ---- 原逻辑: 先运行 dsgdetr ----
        entry = self.dsgdetr(entry)

        im_idx        = entry["im_idx"]
        pair_idx      = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds     = im_idx.size(0)
        num_frames    = len(gt_annotation)

        # 根据 context_fraction 算出 end
        end = int(torch.ceil(torch.tensor(num_frames * context_fraction)).item() - 1)
        end = max(0, min(end, num_frames - 1))

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

        # ======= 滑动窗口逐帧预测开始 ========
        num_future = (num_frames - end - 1)  
        num_objs   = frames_ranges[end + 1] - frames_ranges[end]  
        classA     = self.attention_class_num
        spat_class = self.spatial_class_num
        cont_class = self.contact_class_num

        attn_dist_2d = torch.zeros((num_future * num_objs, classA), device=device)
        spat_dist_2d = torch.zeros((num_future * num_objs, spat_class), device=device)
        cont_dist_2d = torch.zeros((num_future * num_objs, cont_class), device=device)

        n_list = self.get_n_list(num_future, 3)
        current_frame_index = 0

        # 循环开始
        loop_start = time.time()
        max_retries = 3  # 可根据需要调整
        for current_n in n_list:
            # iter_start = time.time()
            step_i = int(context_fraction * 100) + current_frame_index
            known_text, head_line = self.build_known_frames_text(entry, step_i=step_i)
            # print(f"known_text={known_text}")
            fstruct = None
            for attempt_id in range(max_retries):
                future_text = self.llm_anticipator.anticipate_future_frames(
                    known_frames_text=known_text,
                    num_future_frames=current_n,
                    length=512,
                    head_line=head_line  # 如果你有自定义传参
                )
                future_structs = self.llm_anticipator.parse_generated_text_to_graph(future_text)
                print(f"known_text={known_text}")

                if len(future_structs) > 0:
                    # 拿到第一条
                    candidate = future_structs[0]
                    # 检查是否完全空
                    if len(candidate["time"]) > 0 or len(candidate["object_class"]) > 0:
                        fstruct = candidate
                        break  # 成功解析 -> 跳出重试循环
                
                # 如果解析还是空，打印一些提示
                print(f"[Retry {attempt_id+1}/{max_retries}] LLM output is empty, will try again...")

            # 若所有重试都没成功
            if fstruct is None:
                # 你可以选择跳过，或构造个完全空的默认
                fstruct = {
                    'time': [],
                    'object_class': [],
                    'attention_rels': [],
                    'spatial_rels': [],
                    'contact_rels': []
                }
                print("LLM output consistently empty after all retries, fill with empty struct...")

            # 接下来就是 parse_future_struct_and_fill(...) 的逻辑
            attn_dist, spat_dist, cont_dist, filtered_struct = self.parse_future_struct_and_fill(
                fstruct,
                device=device,
                n=current_n
            )
            print(f"filtered_struct={filtered_struct}")

            # print(f"filtered_struct={filtered_struct}")
            # print(f"generated future_text={fstruct}")

            if 'predicted_history' not in entry:
                entry['predicted_history'] = []
            
            entry['predicted_history'].append(filtered_struct)

            for frame_offset in range(current_n):

                start_idx = (current_frame_index + frame_offset) * num_objs
                attn_value = attn_dist[frame_offset] if frame_offset < attn_dist.size(0) else attn_dist[-1]
                spat_value = spat_dist[frame_offset] if frame_offset < spat_dist.size(0) else spat_dist[-1]
                cont_value = cont_dist[frame_offset] if frame_offset < cont_dist.size(0) else cont_dist[-1]

                attn_dist_2d[start_idx:start_idx+num_objs, :] = attn_value.unsqueeze(0).expand(num_objs, -1)
                spat_dist_2d[start_idx:start_idx+num_objs, :] = spat_value.unsqueeze(0).expand(num_objs, -1)
                cont_dist_2d[start_idx:start_idx+num_objs, :] = cont_value.unsqueeze(0).expand(num_objs, -1)
            # print(f"Iteration for current_n={current_n} filling and history update took:", time.time() - iter_start)

            current_frame_index += current_n

        # ======= 滑动窗口预测结束 ========
        pred = {}
        pred["attention_distribution"] = attn_dist_2d
        pred["spatial_distribution"]   = spat_dist_2d
        pred["contacting_distribution"]= cont_dist_2d


        min_idx = torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]])
        max_idx = torch.max(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]) + 1
        repeated_count = num_future

        if self.mode == "predcls":
            pred["scores"] = entry["scores"][min_idx : max_idx].repeat(repeated_count)
            pred["labels"] = entry["labels"][min_idx : max_idx].repeat(repeated_count)
        else:
            pred["pred_scores"] = entry["pred_scores"][min_idx : max_idx].repeat(repeated_count)
            pred["pred_labels"] = entry["pred_labels"][min_idx : max_idx].repeat(repeated_count)



        idx_list = []
        for i in range(num_future):
            idx_list += [i] * num_objs
        pred["im_idx"] = torch.tensor(idx_list, dtype=torch.int32, device=device)



        pair_slice = (pair_idx[frames_ranges[end] : frames_ranges[end + 1]] 
                    - torch.min(pair_idx[frames_ranges[end] : frames_ranges[end + 1]]))
        repeated_slice = pair_slice.unsqueeze(0).repeat(num_future, 1, 1).view(-1, 2)
        mx = num_objs
        offset_im_idx = torch.arange(num_future, device=device).view(num_future, 1).repeat(1, num_objs).view(-1)
        repeated_slice_offset = repeated_slice + mx * offset_im_idx.unsqueeze(-1)
        pred["pair_idx"] = repeated_slice_offset



        max_index = int(pred["pair_idx"].max().item() + 1)
        pred["boxes"] = torch.ones((max_index, 5), device=device) * 0.5

        return end + 1, pred

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
