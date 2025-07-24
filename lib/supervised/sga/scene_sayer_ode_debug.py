"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint

from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierTransformer, GetBoxes
from lib.word_vectors import obj_edge_vectors
import torch.nn.functional as F
ATTN_REL_CLASSES = ['looking_at', 'not_looking_at', 'unsure']
SPAT_REL_CLASSES = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
CONT_REL_CLASSES = ['carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship', 'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']

class STTran(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None,
                 rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None, script_required=False, object_required=False, relation_required=False):

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
        #     self.relationship_head = nn.Linear(d_model + 256, 256)  # 将融合后的特征转换为关系嵌入
        # else:
        #     self.relationship_head = nn.Linear(d_model, 256)  # 不使用脚本时的关系嵌入
        
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

    def forward(self, entry, testing=False):
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
                                    save_path=False):
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
                              relation_required=relation_required)
        self.ctr = 0

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
        for i in range(1, window + 1):
            # masks for final output latents used during loss evaluation
            mask_preds = torch.tensor([], dtype=torch.long, device=frames_ranges.device)
            mask_gt = torch.tensor([], dtype=torch.long, device=frames_ranges.device)
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
                mask_preds = torch.cat((mask_preds, ind1))
                mask_gt = torch.cat((mask_gt, ind2))
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
            batch_times = times_unique[i : i + window + 1]
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
        r"""
        推理函数：
        1. 调用 self.dsgdetr 得到 entry 中的基本信息。
        2. 通过 gt_annotation 中的标注信息构造未来帧的 scene graph 预测输入，
            并调用 build_pred_from_future_frames 构造完整的 pred。
        """
        assert context_fraction > 0

        # 1) 调用 dsgdetr，获取基本信息
        entry = self.dsgdetr(entry)
        im_idx = entry["im_idx"]                # 每个 object 对应的帧编号（可能有重复）
        gt_annotation = entry["gt_annotation"]  # 每帧的标注信息，格式：列表；每帧内部：index 0 可为 subject（或占位），后续为各 object 信息
        num_preds = im_idx.size(0)
        num_frames = len(gt_annotation)         # 总帧数
        device = im_idx.device

        # 2) 计算每帧在 im_idx 中的起止位置（frames_ranges）
        indices = torch.reshape((im_idx[:-1] != im_idx[1:]).nonzero(), (-1,)) + 1
        frames_ranges = torch.cat(
            (
                torch.tensor([0], device=device),
                indices,
                torch.tensor([num_preds], device=device)
            ),
            dim=0
        ).long()

        # 如果相邻帧之间不连续，则插入额外的 ranges 以对齐
        k = frames_ranges.size(0) - 1
        for i in range(k - 1, 0, -1):
            diff = int(im_idx[frames_ranges[i]] - im_idx[frames_ranges[i - 1]])
            if diff > 1:
                frames_ranges = torch.cat(
                    (
                        frames_ranges[:i],
                        torch.tensor([frames_ranges[i]] * (diff - 1), device=device),
                        frames_ranges[i:]
                    )
                )
        # 如果 im_idx[0] > 0，也要在最开始补齐
        if int(im_idx[0]) > 0:
            frames_ranges = torch.cat(
                (
                    torch.tensor([0] * int(im_idx[0]), device=device),
                    frames_ranges
                )
            )

        if frames_ranges.size(0) != num_frames + 1:
            frames_ranges = torch.cat(
                (
                    frames_ranges,
                    torch.tensor([num_preds] * (num_frames + 1 - frames_ranges.size(0)), device=device)
                )
            )

        # 3) 计算观测帧的结束帧（end），即仅观测到第 end 帧
        end = int(np.ceil(num_frames * context_fraction) - 1)
        while end > 0 and frames_ranges[end] == frames_ranges[end + 1]:
            end -= 1

        # 如果观测帧已为最后一帧或该帧无对象，则直接返回空预测
        if end == num_frames - 1 or frames_ranges[end] == frames_ranges[end + 1]:
            return num_frames, {}

        # ---------------------------
        # 4) 根据 gt_annotation 构造未来帧（f in [end+1, num_frames-1]）的输入，
        #    用于生成 scene graph 的预测结果。
        future_num_frames = num_frames - (end + 1)
        im_idx_list = []   # 记录每帧中对象的数量
        labels_list = []   # 记录每帧中各对象的标签（subject 固定为 1，不在此处传入）
        attn_list = []     # 记录每帧中各对象的 attention one-hot 张量
        spat_list = []     # 记录 spatial one-hot 张量
        cont_list = []     # 记录 contacting one-hot 张量

        for f in range(end + 1, num_frames):
            frame_anno = gt_annotation[f]  # 当前帧标注，格式为列表：[subject, obj1, obj2, ...]
            num_objs = len(frame_anno) - 1  # 当前帧中 object 的数量
            im_idx_list.append(num_objs)

            frame_obj_labels = []  # 用于存储当前帧中各 object 的类别标签
            frame_attn = []        # 存储当前帧中各 object 的 attention one-hot 向量
            frame_spat = []        # 存储 spatial one-hot 向量
            frame_cont = []        # 存储 contacting one-hot 向量

            for m in range(num_objs):
                obj_info = frame_anno[m + 1]  # 注意：frame_anno[0] 可能为 subject（或占位），对象从 index 1 开始
                # 获取 object 标签
                frame_obj_labels.append(obj_info["class"])

                # 构造 attention one-hot 向量（类别数为 3）
                att_tensor = obj_info["attention_relationship"]
                att_onehot = torch.zeros(3, device=device).float()
                for i, val in enumerate(att_tensor):
                    idx = int(val.item())
                    att_onehot[idx] = 1 - 0.1 * i
                frame_attn.append(att_onehot)

                # 构造 spatial one-hot 向量（类别数为 6）
                spat_tensor = obj_info["spatial_relationship"]
                spat_onehot = torch.zeros(6, device=device).float()
                for i, val in enumerate(spat_tensor):
                    idx = int(val.item())
                    spat_onehot[idx] = 1 - 0.1 * i
                frame_spat.append(spat_onehot)

                # 构造 contacting one-hot 向量（类别数为 17）
                cont_tensor = obj_info["contacting_relationship"]
                cont_onehot = torch.zeros(17, device=device).float()
                for i, val in enumerate(cont_tensor):
                    idx = int(val.item())
                    cont_onehot[idx] = 1 - 0.1 * i
                frame_cont.append(cont_onehot)

            # 若当前帧有对象，则将列表转换为 tensor；否则构造空 tensor
            if num_objs > 0:
                frame_attn_tensor = torch.stack(frame_attn, dim=0)   # [num_objs, 3]
                frame_spat_tensor = torch.stack(frame_spat, dim=0)     # [num_objs, 6]
                frame_cont_tensor = torch.stack(frame_cont, dim=0)     # [num_objs, 17]
            else:
                frame_attn_tensor = torch.empty((0, 3), device=device)
                frame_spat_tensor = torch.empty((0, 6), device=device)
                frame_cont_tensor = torch.empty((0, 17), device=device)

            labels_list.append(frame_obj_labels)
            attn_list.append(frame_attn_tensor)
            spat_list.append(frame_spat_tensor)
            cont_list.append(frame_cont_tensor)

        # 5) 利用封装函数构造完整的 pred
        print(labels_list)
        pred = self.build_pred_from_future_frames(future_num_frames, im_idx_list, labels_list,
                                            attn_list, spat_list, cont_list, device)

        return end + 1, pred

    def build_pred_from_future_frames(self, num_future_frames, im_idx_list, labels_list,
                                        attn_list, spat_list, cont_list, device):
        r"""
        根据未来帧的信息构造 scene graph 预测结果。

        输入参数：
            num_future_frames (int): 未来预测的帧数。
            im_idx_list (list[int]): 长度为 num_future_frames 的列表，每个元素为该帧的对象数量。
            labels_list (list[list[int]]): 长度为 num_future_frames 的列表，每个元素为该帧中所有对象的类别标签。
                                        注意：subject 的标签固定为 1，会自动添加在每一帧的首位，
                                        因此此处只需给出各帧中 object 的标签。
            attn_list (list[torch.Tensor]): 长度为 num_future_frames 的列表，每个 tensor 的形状为 [num_objs, 3]，
                                            表示当前帧中各对象的 attention one-hot 分布。
            spat_list (list[torch.Tensor]): 长度为 num_future_frames 的列表，每个 tensor 的形状为 [num_objs, 6]，
                                            表示 spatial one-hot 分布。
            cont_list (list[torch.Tensor]): 长度为 num_future_frames 的列表，每个 tensor 的形状为 [num_objs, 17]，
                                            表示 contacting one-hot 分布。
            device (torch.device): 指定生成 tensor 的设备。

        返回：
            pred (dict): 包含以下 key：
                - "labels": 1D tensor, 所有节点的标签（每帧第一个节点为 subject，固定为 1，其余为 object 标签）。
                - "scores": 1D tensor, 每个节点的分数（这里统一设为 1.0）。
                - "im_idx": 1D tensor, 长度等于所有未来帧中对象的总数，每个元素表示该对象所属的未来帧编号（从 0 开始）。
                - "pair_idx": 2D tensor, 每一行为 [subject_idx, object_idx]，表示每一帧中 subject 与各 object 的关联关系。
                - "boxes": 2D tensor, 占位符 boxes，形状为 [(num_future_frames + 总对象数), 5]，均初始化为 0.5。
                - "attention_distribution": 2D tensor, 将各帧中对象的 attention 分布按行拼接，形状为 [总对象数, 3]。
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
            # 构造当前帧中 subject 与每个 object 的 pair_idx，
            # subject 的全局索引为 current_idx，object 分别为 current_idx+1, current_idx+2, …, current_idx+num_objs
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

        # 拼接各未来帧中对象的 relationship distribution
        total_objects = sum(im_idx_list)
        if total_objects > 0:
            pred["attention_distribution"] = torch.cat(attn_list, dim=0)  # shape: [总对象数, 3]
            pred["spatial_distribution"] = torch.cat(spat_list, dim=0)      # shape: [总对象数, 6]
            pred["contacting_distribution"] = torch.cat(cont_list, dim=0)     # shape: [总对象数, 17]
        else:
            pred["attention_distribution"] = torch.empty((0, 3), device=device)
            pred["spatial_distribution"] = torch.empty((0, 6), device=device)
            pred["contacting_distribution"] = torch.empty((0, 17), device=device)

        return pred
    