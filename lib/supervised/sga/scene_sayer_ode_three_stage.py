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
from llama_SGA.SGA_stage_0_data import SceneGraphAllocator as SceneGraphFineTuner0
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
def time_encoding(frame_id, fps=24):
    # 将帧ID转换为秒数
    time_in_seconds = frame_id / fps
    # 创建多尺度时间表示
    return {
        "absolute_time": time_in_seconds,
        "minute_mark": int(time_in_seconds // 60),
        "second_mark": time_in_seconds % 60
    }
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
    
    def print_indices(self, entry, threshold=None):
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]  # 主体索引
        obj_class  = entry['pred_labels'][entry['pair_idx'][:, 1]]  # 客体索引
        attn_rel_indices = torch.argmax(entry["attention_distribution"], dim=1) 
        if threshold is not None:
            # 对spatial和contact使用阈值选择多标签，但最多取2个
            spatial_probs = torch.sigmoid(entry["spatial_distribution"])
            spaitial_rel_indices = []
            for i in range(spatial_probs.size(0)):
                # 获取所有大于阈值的索引
                indices = torch.where(spatial_probs[i] > threshold)[0]
                
                if len(indices) == 0:  # 如果没有高过阈值的，则选择最高分的一个
                    indices = torch.topk(spatial_probs[i], 1)[1]
                elif len(indices) > 2:  # 如果高过阈值的超过2个，只保留分数最高的2个
                    # 获取这些索引对应的概率值
                    values = spatial_probs[i][indices]
                    # 对这些概率值进行排序并保留前2个
                    _, top_idx = torch.topk(values, 2)
                    indices = indices[top_idx]
                    
                spaitial_rel_indices.append(indices)
            
            # 对contact关系使用阈值选择多标签，但最多取2个
            contact_probs = torch.sigmoid(entry["contacting_distribution"])
            contacting_rel_indices = []
            for i in range(contact_probs.size(0)):
                # 获取所有大于阈值的索引
                indices = torch.where(contact_probs[i] > threshold)[0]
                
                if len(indices) == 0:  # 如果没有高过阈值的，则选择最高分的一个
                    indices = torch.topk(contact_probs[i], 1)[1]
                elif len(indices) > 2:  # 如果高过阈值的超过2个，只保留分数最高的2个
                    # 获取这些索引对应的概率值
                    values = contact_probs[i][indices]
                    # 对这些概率值进行排序并保留前2个
                    _, top_idx = torch.topk(values, 2)
                    indices = indices[top_idx]
                    
                contacting_rel_indices.append(indices)
        else:
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
                 model_path_stage0=None,
                 lora_path_stage0=None,
                 lora_path_stage1=None,
                 lora_path_stage2=None,
                 use_fusion=False,
                 save_path=False,
                 use_gt_anno=False,
                 threshold=None,
                 stage1_tokens=256,
                 length_of_segments=None,
                 not_use_merge=False,
                 use_stage0=False,
                 ):
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
        self.use_gt_anno = use_gt_anno
        self.threshold = threshold
        self.stage1_tokens = stage1_tokens
        self.length_of_segments = length_of_segments
        self.not_use_merge = not_use_merge
        self.use_stage0 = use_stage0

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
        self.stage0_anticipator = SceneGraphFineTuner0(
            model_path=model_path_stage0,  # llama_path
            ckpt_path=lora_path_stage0,  # lora_path_stage1
            phase="eval",
            local_rank=0,  # 添加默认值
            world_size=1,  # 添加默认值
            object_classes=obj_classes,
        )

        # 初始化 Stage 1
        self.stage1_anticipator = SceneGraphFineTuner(
            model_path=llama_path,
            ckpt_path=lora_path_stage1,
            phase="eval",
            local_rank=0,  # 添加默认值
            world_size=1,  # 添加默认值
            object_classes = obj_classes,
        )

        # # 初始化 Stage 2
        # self.stage2_allocator = SceneGraphAllocator(
        #     model_path=llama_path,
        #     ckpt_path=lora_path_stage2,
        #     phase="eval",
        #     local_rank=0,  # 添加默认值
        #     world_size=1,  # 添加默认值
        # )

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
        subj_class, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices = self.dsgdetr.print_indices(entry, threshold=self.threshold)
        if self.use_gt_anno:
            observed_anno = gt_annotation[:end + 1]
        else:
            observed_anno = self.build_frames_annotation(im_idx, obj_class, attn_rel_indices, sp_rel_indices, cont_rel_indices, times)[:end + 1]
        future_anno = gt_annotation[end + 1:]
        num_future = num_frames - end - 1

        if self.length_of_segments is not None and len(observed_anno) > self.length_of_segments:
            observed_anno = observed_anno[-self.length_of_segments:]

        if self.not_use_merge:
            obs_segments = self._no_merge_frames_for_objects_inference(observed_anno)
        else:
            obs_segments = self._merge_frames_for_objects_inference(observed_anno)
        obs_by_obj = self._group_segments_by_object_inference(obs_segments)
        observed_objects = set()
        for frame_data in observed_anno:
            for obj in frame_data[1:]:  # 从索引1开始遍历(跳过第0个frame meta信息)
                if 'class' in obj and 0 <= obj['class'] < len(self.obj_classes):
                    obj_class = self.obj_classes[obj['class']]
                    observed_objects.add(obj_class)

        max_tokens = (9-self.stage1_tokens/256)*300
        stage_0_max_tokens = (self.stage1_tokens/256-1)*256
        # # 获取最后一帧出现的对象
        # ===========================和stage0互斥===========================
        if self.use_stage0 == False:
            end_frame_objects = []  # 改用list而不是set来保持顺序
            if len(observed_anno) > 0:
                last_frame = observed_anno[-1]
                for obj in last_frame[1:]:  # 从索引1开始遍历(跳过第0个frame meta信息)
                    if 'class' in obj and 0 <= obj['class'] < len(self.obj_classes):
                        obj_class = self.obj_classes[obj['class']]
                        if obj_class not in end_frame_objects:  # 仍然需要去重，但保持首次出现的顺序
                            end_frame_objects.append(obj_class)
            all_objects = end_frame_objects

            future_frames = []
            for frame_data in future_anno:
                frame_num = self._extract_frame_number(frame_data[0]['frame'])
                future_frames.append(frame_num)
            
            future_objects_by_frame = {}
            for frame in future_frames:
                future_objects_by_frame[frame] = end_frame_objects.copy()

        # ===========================end===========================
        # ===========================stage0===========================
        else:
        # Step 4: 使用 Stage 0 预测未来帧的物体
            future_frames = []
            for frame_data in future_anno:
                frame_num = self._extract_frame_number(frame_data[0]['frame'])
                future_frames.append(frame_num)
            
            use_timestamp = False
            
            observed_text = self._aggregate_frames_annotation(observed_anno, max_tokens=max_tokens, timestamps=use_timestamp)
            # observed_text = self._aggregate_scene_graph_ctrl_len(observed_anno)

            stage0_prompt = self.stage0_anticipator.build_prompt(
                observed_text=observed_text,
                future_frames=future_frames,
                observed_anno = observed_anno,
                max_length=max_tokens,
                include_timestamps=use_timestamp
            )

            max_attempts = 2
            if self.length_of_segments is not None:
                self.stage1_tokens = 256
            for attempt in range(max_attempts):
                stage0_generated = self.stage0_anticipator.generate_text(
                    prompts=[stage0_prompt],
                    max_new_tokens=self.stage1_tokens,
                    temperature=0.7, #0.
                    do_sample=True,
                    top_p=0.4, #0.4
                )[0]
                stage0_generated_text = stage0_generated.replace(stage0_prompt, "")
                # print(f"Stage 0 生成的文本（尝试 {attempt+1}/{max_attempts}）：{stage0_generated_text}")
                # breakpoint()

                future_objects_by_frame = self.stage0_anticipator.parse_generated_objects(stage0_generated_text, time=use_timestamp)



                future_objects_by_frame = self.ensure_frames_consistency(future_objects_by_frame, future_frames)
                
                # 检查future_objects_by_frame是否为空
                is_empty = False
                for frame, objects in future_objects_by_frame.items():
                    if objects==[]:  # 如果任何帧有有效对象
                        is_empty = True
                        break
                
                if not is_empty:
                    # print(f"成功生成非空的future_objects_by_frame")
                    break
                elif attempt < max_attempts - 1:
                    print(f"警告：生成的future_objects_by_frame为空，正在重新生成（尝试 {attempt+1}/{max_attempts}）")
                else:
                    print(f"警告：多次尝试后，future_objects_by_frame仍然为空")

            # 如果所有尝试后仍然为空，使用最后一个观测帧的对象或默认对象
            if is_empty and len(observed_anno) > 0:
                print("使用最后一个观测帧的对象作为未来帧物体")
                last_frame = observed_anno[-1]
                last_frame_objects = []
                for obj in last_frame[1:]:  # 从索引1开始遍历(跳过第0个frame meta信息)
                    if 'class' in obj and 0 <= obj['class'] < len(self.obj_classes):
                        obj_class = self.obj_classes[obj['class']]
                        if obj_class not in last_frame_objects:
                            last_frame_objects.append(obj_class)
                
                # 用最后观测到的对象填充所有未来帧
                for frame in future_frames:
                    future_objects_by_frame[frame] = last_frame_objects.copy()
            
        # print(f"Stage 0 解析的未来帧物体：{future_objects_by_frame}")

        # 补充每个未来帧的对象列表，确保包含最后一个观测帧的所有对象
        # for frame in future_frames:
        #     if frame in future_objects_by_frame:
        #         future_objects_by_frame[frame] = list(set(end_frame_objects) | set(future_objects_by_frame.get(frame, [])))
        #     else:
        #         future_objects_by_frame[frame] = end_frame_objects.copy()

        # # ===========================end===========================
        # # 构建每个物体的未来帧映射
        obj_to_future_frames = defaultdict(list)
        for frame_idx, objects in future_objects_by_frame.items():
            for obj in objects:
                if obj in self.obj_classes:
                    if frame_idx not in obj_to_future_frames[obj]:
                        obj_to_future_frames[obj].append(frame_idx)
        
        # print(f"未来帧物体预测：{obj_to_future_frames}")
        all_objects = list(obj_to_future_frames.keys())
        if not all_objects:
            return num_frames, {}
    

        

        # Step 5: 使用 Stage 1 预测所有对象的未来 Scene Graph
        
        max_batch_size = 10  # 每批最多处理3个物体，避免OOM
        
        # 分批处理所有物体
        all_results = []
        all_obj_list = []
        all_obj_num_future = []
        
        # 将对象分成多个批次
        for i in range(0, len(all_objects), max_batch_size):
            batch_objects = all_objects[i:i + max_batch_size]
            
            # 为当前批次准备输入
            batch_prompts = []
            batch_obj_list = []
            batch_obj_num_future = []
            
            for obj_cls in batch_objects:
                obs_obj_segments = obs_by_obj.get(obj_cls, [])
                if not obs_obj_segments:
                    continue
                observed_text = self._build_text_from_segments_for_object(obs_obj_segments, obj_cls, observed=True, max_tokens=max_tokens).split("\n")
                num_future_obj = len(obj_to_future_frames[obj_cls])
                full_prompt = self.stage1_anticipator.build_prompt_for_scene_graph(
                    observed_segments=observed_text,
                    object_class=obj_cls,
                    relationship_categories=self.relationship_categories,
                    num_future_frames=num_future_obj,
                    future_frames=obj_to_future_frames[obj_cls]
                )
                batch_prompts.append(full_prompt)
                batch_obj_list.append(obj_cls)
                batch_obj_num_future.append(num_future_obj)
            
            if not batch_prompts:
                continue
                
            # 为当前批次生成文本
            # print(f"处理批次 {i//max_batch_size + 1}/{(len(all_objects) + max_batch_size - 1)//max_batch_size}，包含 {len(batch_prompts)} 个对象")
            
            max_attempts = 2
            batch_results = [None] * len(batch_prompts)
            pending_indices = list(range(len(batch_prompts)))
            attempt = 0
            
            while attempt < max_attempts and pending_indices:
                current_batch_prompts = [batch_prompts[i] for i in pending_indices]
                batch_generated = self.stage1_anticipator.generate_text(
                    prompts=current_batch_prompts,
                    max_new_tokens=self.stage1_tokens+300,
                    temperature=0.7,
                    top_p=0.4
                )
                if isinstance(batch_generated, str):
                    batch_generated = [batch_generated]
                    
                new_pending = []
                for idx_in_batch, gen_text in enumerate(batch_generated):
                    i_obj = pending_indices[idx_in_batch]
                    gen_text = gen_text.replace(batch_prompts[i_obj], "")
                    lines_raw = self._split_generated_to_lines(gen_text)
                    lines_parsed, line_id = self.extract_future_segments(lines_raw, obj_to_future_frames[batch_obj_list[i_obj]])
                    needed = batch_obj_num_future[i_obj]
                    if len(lines_parsed) >= needed:
                        lines_use = lines_parsed[:needed]
                    elif len(lines_parsed) < needed:
                        lines_use = lines_parsed[:]
                        while len(lines_use) < needed:
                            lines_use.append(lines_use[-1] if lines_use else "")
                    if any(line == '' for line in lines_use):
                        print(f"Prompt {i_obj} 尝试 {attempt + 1} 次后仍包含空字符串，准备重试...")
                        new_pending.append(i_obj)
                    else:
                        batch_results[i_obj] = lines_use
                
                pending_indices = new_pending
                attempt += 1
            
            if any(r is None for r in batch_results):
                print(f"警告：批次 {i//max_batch_size + 1} 中有生成失败的结果，跳过这些对象")
                # 过滤掉失败的结果
                valid_results = []
                valid_obj_list = []
                valid_obj_num_future = []
                for j, res in enumerate(batch_results):
                    if res is not None:
                        valid_results.append(res)
                        valid_obj_list.append(batch_obj_list[j])
                        valid_obj_num_future.append(batch_obj_num_future[j])
                batch_results = valid_results
                batch_obj_list = valid_obj_list
                batch_obj_num_future = valid_obj_num_future
            
            # 收集当前批次的结果
            all_results.extend(batch_results)
            all_obj_list.extend(batch_obj_list)
            all_obj_num_future.extend(batch_obj_num_future)

        if not all_obj_list:
            print("警告：经过 3 次尝试后，仍存在生成失败的结果。")
            return num_frames, {}

        # 按原顺序拼接生成结果
        lines_batch = []
        obj_line_ranges = []
        running_idx = 0
        for res in all_results:
            start_idx = running_idx
            lines_batch.extend(res)
            running_idx += len(res)
            end_idx = running_idx
            obj_line_ranges.append((start_idx, end_idx))
        
        # print(f"Stage 1 生成的文本：{lines_batch}")

        # Step 7: 处理分配后的 Scene Graph 并计算分布
        if self.use_classify_head:
            # dist_mat_all = self.classify_generated_text_for_object(assigned_scene_graphs, None, device=device)
            dist_mat_all = self.classify_generated_text_for_object(lines_batch, None, device=device, parse=True)
        else:
            dist_mat_all = self.classify_generated_text_for_object_wo_classification_head(lines_batch, None)

        # 构建 distribution_dict
        distribution_dict = {}
        frame_to_obj_idx = defaultdict(list)
        for i_obj, obj_cls in enumerate(all_obj_list):
            start_idx, end_idx = obj_line_ranges[i_obj]
            dist_mat_obj = dist_mat_all[start_idx:end_idx, :]
            M = dist_mat_obj.size(0)
            obj_future_frames = obj_to_future_frames[obj_cls]
            if M < len(obj_future_frames):
                last_row = dist_mat_obj[-1].unsqueeze(0) if M > 0 else torch.zeros(1, dist_mat_all.size(1), device=device)
                pad_rows = last_row.repeat(len(obj_future_frames) - M, 1)
                dist_mat_obj = torch.cat([dist_mat_obj, pad_rows], dim=0)
            elif M > len(obj_future_frames):
                dist_mat_obj = dist_mat_obj[:len(obj_future_frames), :]
            distribution_dict[obj_cls] = {
                "dist": dist_mat_obj,
                "frames": torch.tensor(obj_future_frames, device=device)
            }
            for frame_idx in obj_future_frames:
                frame_to_obj_idx[frame_idx].append(i_obj)


        # Step 8: 生成关系分布和最终预测
        attn_mat, spat_mat, cont_mat = self.generate_relationship_distributions(
                                        distribution_dict, future_frames, all_obj_list, frame_to_obj_idx, device
                                    )

        # 准备 build_pred_from_future_frames 的输入
        im_idx_list = [len(frame_to_obj_idx[frame]) for frame in future_frames]
        labels_list = [[self.obj_classes.index(all_obj_list[i]) for i in frame_to_obj_idx[frame]] for frame in future_frames]

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

    def ensure_frames_consistency(self, future_objects_by_frame, future_frames):
        """
        确保future_objects_by_frame字典包含且仅包含future_frames中的所有帧。
        - 对于缺少的帧，从最近的前一帧复制物体列表
        - 对于多余的帧（不在future_frames中），将其删除
        
        参数:
            future_objects_by_frame (dict): {frame_idx -> list of objects}，当前预测的物体字典
            future_frames (list): 需要预测的帧编号列表，按升序排列
            
        返回:
            dict: 修正后的物体字典，确保包含且仅包含future_frames中的帧
        """
        if not future_frames:
            return {}
        
        # 按帧编号排序future_frames
        sorted_frames = sorted(future_frames)
        
        # 创建新字典存储结果
        corrected_dict = {}
        
        # 处理每一帧
        for frame_idx in sorted_frames:
            if frame_idx in future_objects_by_frame:
                # 如果帧存在，直接复制
                corrected_dict[frame_idx] = future_objects_by_frame[frame_idx]
            else:
                # 如果帧不存在，寻找最近的前一帧
                prev_objects = []
                # 查找已处理的帧中小于当前帧的最大帧
                prev_frames = [f for f in corrected_dict.keys() if f < frame_idx]
                if prev_frames:
                    prev_frame = max(prev_frames)
                    prev_objects = corrected_dict[prev_frame]
                # 如果没有前一帧但有后一帧，也可以使用它（这是一个备选方案）
                elif future_objects_by_frame:
                    # 找到future_objects_by_frame中最近的一帧
                    available_frames = list(future_objects_by_frame.keys())
                    nearest_frame = min(available_frames, key=lambda f: abs(f - frame_idx))
                    prev_objects = future_objects_by_frame[nearest_frame]
                
                # 将结果添加到修正后的字典
                corrected_dict[frame_idx] = prev_objects
        
        return corrected_dict
        
    def _build_observed_text_from_segments(self, segments, include_time=False):
        lines = []
        for seg in segments:
            obj_cls = seg["object_class"]
            line = self._build_text_from_segments_for_object([seg], obj_cls, observed=True, include_time=include_time)
            lines.append(line)
        return "\n".join(lines) + "\n"
    
    def build_frames_annotation(
        self,
        im_idx: torch.Tensor,
        obj_class: torch.Tensor,
        attn_rel_indices: torch.Tensor,
        spatial_rel_indices,  # 可能是list或tensor
        contacting_rel_indices,  # 可能是list或tensor
        time: list
    ) -> list:
        """将输入转化为 frames_annotation 格式。"""
        # 确保im_idx和obj_class是一维张量
        assert im_idx.dim() == 1, "im_idx 必须是一维张量"
        assert obj_class.dim() == 1, "obj_class 必须是一维张量"
        
        # 将张量转为列表
        im_idx = im_idx.tolist()
        obj_class = obj_class.tolist()
        
        # 处理不同类型的关系索引
        if isinstance(attn_rel_indices, torch.Tensor):
            attn_rel_indices = attn_rel_indices.tolist()
        
        # 按帧索引分组对象
        frames_dict = defaultdict(list)
        for i in range(len(im_idx)):
            frame_id = time[im_idx[i]]
            
            # 处理不同类型的空间关系
            if isinstance(spatial_rel_indices, list):
                # 如果是列表，获取第i个元素并转为列表
                if isinstance(spatial_rel_indices[i], torch.Tensor):
                    spat_rel = spatial_rel_indices[i].tolist()
                else:
                    spat_rel = spatial_rel_indices[i]
            else:
                # 如果是张量，获取第i个元素
                spat_rel = [spatial_rel_indices[i].item() if isinstance(spatial_rel_indices, torch.Tensor) 
                        else spatial_rel_indices[i]]
            
            # 处理不同类型的接触关系
            if isinstance(contacting_rel_indices, list):
                # 如果是列表，获取第i个元素并转为列表
                if isinstance(contacting_rel_indices[i], torch.Tensor):
                    cont_rel = contacting_rel_indices[i].tolist()
                else:
                    cont_rel = contacting_rel_indices[i]
            else:
                # 如果是张量，获取第i个元素
                cont_rel = [contacting_rel_indices[i].item() if isinstance(contacting_rel_indices, torch.Tensor) 
                        else contacting_rel_indices[i]]
            
            obj_dict = {
                'class': obj_class[i],
                'attention_relationship': [attn_rel_indices[i]],
                'spatial_relationship': spat_rel,
                'contacting_relationship': cont_rel
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

    def generate_relationship_distributions(self, distribution_dict, future_frames, obj_list, frame_to_obj_idx, device):
        """
        Generate attention, spatial, and contacting relationship distributions in a frame-first order,
        including only objects present in each frame.

        Parameters:
        - distribution_dict: dict[obj_cls -> {"dist": Tensor[K, 26], "frames": Tensor[K]}]
        - "dist": Distribution tensor of shape [K, 26] for each object, where K is the number of frames it appears in.
        - "frames": Tensor of frame indices corresponding to each row in "dist".
        - future_frames: list of frame indices to predict (e.g., [end+1, end+2, ...]).
        - obj_list: list of object classes corresponding to indices in frame_to_obj_idx.
        - frame_to_obj_idx: dict[frame_idx -> list of i_obj], mapping each frame to indices of objects in obj_list.

        Returns:
        - attn_tensor: Tensor of shape [total_objects, attention_class_num]
        - spat_tensor: Tensor of shape [total_objects, spatial_class_num]
        - cont_tensor: Tensor of shape [total_objects, contact_class_num]
        where total_objects is the sum of object counts across all future frames.
        """
        attn_list = []
        spat_list = []
        cont_list = []

        # Iterate over each future frame
        for frame_idx in future_frames:
            # Get indices of objects present in this frame
            obj_indices = frame_to_obj_idx[frame_idx]
            for i_obj in obj_indices:
                obj_cls = obj_list[i_obj]
                dist_info = distribution_dict[obj_cls]
                frames = dist_info["frames"]  # Tensor[K] of frame indices
                dist_mat = dist_info["dist"]  # Tensor[K, 26] of distributions

                # Find the row index k where frames[k] matches frame_idx
                mask = (frames == frame_idx)
                if mask.any():
                    k = mask.nonzero(as_tuple=True)[0][0].item()
                    row_26d = dist_mat[k]
                else:
                    raise ValueError(f"Object {obj_cls} listed in frame {frame_idx} but not found in its frames tensor.")

                # Split the 26D distribution into attention, spatial, and contacting parts
                attn_vec = row_26d[:self.attention_class_num]
                spat_vec = row_26d[self.attention_class_num:self.attention_class_num + self.spatial_class_num]
                cont_vec = row_26d[self.attention_class_num + self.spatial_class_num:]
                attn_list.append(attn_vec)
                spat_list.append(spat_vec)
                cont_list.append(cont_vec)

        # Stack into tensors
        attn_tensor = torch.stack(attn_list, dim=0) if attn_list else torch.empty((0, self.attention_class_num), device=device)
        spat_tensor = torch.stack(spat_list, dim=0) if spat_list else torch.empty((0, self.spatial_class_num), device=device)
        cont_tensor = torch.stack(cont_list, dim=0) if cont_list else torch.empty((0, self.contact_class_num), device=device)

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
    
    def extract_future_segments(self, lines, future_num):
        """
        从给定的 lines 列表中解析并提取符合格式的行，同时返回对应的Frame ID。
        支持两种格式：
        1. 新格式: "attention: ... spatial: ... contact: ..."
        2. 旧格式: "Person attention to the object: ... the object located relative to person: ... Person contact with the object: ..."
        
        参数:
        lines (List[str]): 整段生成文本split("\n")得到的行列表。
        future_num (List[int]): 需要预测的所有帧号列表。

        返回:
        tuple: (future_lines, frame_ids)
            future_lines (List[str]): 处理后的未来帧文本行
            frame_ids (List[int]): 对应的Frame ID
        """
        # 初始化
        new_pattern = _PATTERN_LINE  # 新格式模式
        frame_pattern = re.compile(r"Frame\s+(\d+)(?:\.\.(\d+))?:")
        
        # 旧格式模式 - 分别匹配三个关系部分
        attn_pattern = re.compile(r'Person attention to the object:\s*([^,]*?)(?=,)', re.IGNORECASE)
        spat_pattern = re.compile(r'the object located relative to person:\s*([^,]*?)(?=,)', re.IGNORECASE)
        cont_pattern = re.compile(r'Person contact with the object:\s*([^,\.]*)', re.IGNORECASE)
        obj_pattern = re.compile(r'object:\s*([^P]*?)(?=Person|$)', re.IGNORECASE)

        # 第一步：提取所有匹配的行及其帧号，并进行格式转换
        raw_lines = []
        raw_frame_ids = []

        for line in lines:
            line = line.strip()
            frame_match = frame_pattern.search(line)
            if not frame_match:
                continue  # 没有帧号，跳过
                
            frame_id = int(frame_match.group(1))
            
            # 检查是否是新格式
            if new_pattern.search(line):
                # 已经是新格式，直接添加
                raw_lines.append(line)
                raw_frame_ids.append(frame_id)
                continue
            
            # 检查是否是旧格式
            attn_match = attn_pattern.search(line)
            spat_match = spat_pattern.search(line)
            cont_match = cont_pattern.search(line)
            
            # 如果找到了旧格式的任何部分，进行转换
            if attn_match or spat_match or cont_match:
                # 提取值，如果没找到则使用 "None"
                attn_value = attn_match.group(1).strip() if attn_match else "None"
                spat_value = spat_match.group(1).strip() if spat_match else "None"
                cont_value = cont_match.group(1).strip() if cont_match else "None"
                
                # 提取对象部分
                obj_match = obj_pattern.search(line)
                obj_text = obj_match.group(1).strip() if obj_match else ""
                
                # 构建新格式的行
                frame_part = frame_match.group(0)  # "Frame N:"
                new_line = f"{frame_part} object: {obj_text} attention: {attn_value}, spatial: {spat_value}, contact: {cont_value}."
                
                raw_lines.append(new_line)
                raw_frame_ids.append(frame_id)
        
        # 第二步：过滤掉不在future_num中的帧
        filtered_lines = []
        filtered_frame_ids = []
        for i, frame_id in enumerate(raw_frame_ids):
            if frame_id in future_num:  # 保留在future_num中的帧
                filtered_lines.append(raw_lines[i])
                filtered_frame_ids.append(frame_id)
        
        # 第三步：填充缺失的帧（对future_num中存在但结果中没有的帧）
        final_lines = []
        final_frame_ids = []
        
        for target_frame in future_num:
            if target_frame in filtered_frame_ids:
                # 如果该帧已存在，直接添加
                idx = filtered_frame_ids.index(target_frame)
                final_lines.append(filtered_lines[idx])
                final_frame_ids.append(target_frame)
            else:
                # 如果该帧缺失，寻找前一个有效帧
                prev_frame = None
                prev_line = None
                
                for i, frame_id in enumerate(final_frame_ids):
                    if frame_id < target_frame:
                        prev_frame = frame_id
                        prev_line = final_lines[i]
                
                # 如果找到前一帧，复制并修改帧号
                if prev_line:
                    # 替换Frame标记部分
                    new_line = re.sub(r"Frame\s+\d+(?:\.\.\d+)?:", f"Frame {target_frame}:", prev_line)
                    final_lines.append(new_line)
                    final_frame_ids.append(target_frame)
                else:
                    # 如果没有前一帧，但有任何帧，使用第一个可用帧
                    if filtered_lines:
                        first_line = filtered_lines[0]
                        new_line = re.sub(r"Frame\s+\d+(?:\.\.\d+)?:", f"Frame {target_frame}:", first_line)
                        final_lines.append(new_line)
                        final_frame_ids.append(target_frame)
        
        return final_lines, final_frame_ids
    
    def _extract_frame_number(self, frame_info):
        """从帧信息中提取帧号"""
        try:
            return int(frame_info.split('/')[-1].split('.')[0])
        except:
            return 0
    
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
    
    def _aggregate_frames_annotation(self, frames_annotation, max_tokens=None, timestamps=True):
        """聚合帧的 scene graph，合并相同帧并标注 Frame ID 区间，
        并在总 token 数超出 max_tokens 时，从最早的区间开始一次性丢弃。"""
        # 1. 按照连续相同 scene graph 合并成 intervals
        fps=24
        intervals = []
        start_idx = 0
        n = len(frames_annotation)
        while start_idx < n:
            end_idx = start_idx
            while (end_idx + 1 < n and 
                self._compare_frame_data(frames_annotation[end_idx],
                                            frames_annotation[end_idx + 1])):
                end_idx += 1
            fr_s = self._extract_frame_number(
                frames_annotation[start_idx][0].get('frame', '0')
            )
            fr_e = self._extract_frame_number(
                frames_annotation[end_idx][0].get('frame', '0')
            )
            intervals.append((fr_s, fr_e, frames_annotation[start_idx]))
            start_idx = end_idx + 1

        # 2. 为每个 interval 构建文本块，并用 tokenizer 计算 token 数
        tokenizer = self.stage1_anticipator.tokenizer
        interval_texts = []
        token_counts = []
        for fr_s, fr_e, frame_data in intervals:
            lines = []
            # header
            if timestamps:
                # 添加时间戳
                time_start = time_encoding(fr_s, fps)
                time_end = time_encoding(fr_e, fps)
                if fr_s == fr_e:
                    lines.append(f"Frame {fr_s} [T={time_start['absolute_time']:.2f}s]:")
                else:
                    lines.append(f"Frame {fr_s}-{fr_e} [T={time_start['absolute_time']:.2f}s-{time_end['absolute_time']:.2f}s]:")
            else:
                # 原始格式
                if fr_s == fr_e:
                    lines.append(f"Frame {fr_s}:")
                else:
                    lines.append(f"Frame {fr_s}-{fr_e}:")

            # 对象行
            for obj in frame_data[1:]:
                cls_idx = obj.get('class', -1)
                name = (self.obj_classes[cls_idx]
                        if 0 <= cls_idx < len(self.obj_classes)
                        else "unknown")
                attn = obj.get('attention_relationship', [])
                spat = obj.get('spatial_relationship', [])
                cont = obj.get('contacting_relationship', [])
                if hasattr(spat, 'tolist'): spat = spat.tolist()
                if hasattr(cont, 'tolist'): cont = cont.tolist()
                attn_str = (",".join(self.attn_rel_classes[i].replace('_', ' ') for i in attn
                                    if 0 <= i < len(self.attn_rel_classes))
                            or "None")
                spat_str = (",".join(self.spat_rel_classes[i].replace('_', ' ') for i in spat
                                    if 0 <= i < len(self.spat_rel_classes))
                            or "None")
                cont_str = (",".join(self.cont_rel_classes[i].replace('_', ' ') for i in cont
                                    if 0 <= i < len(self.cont_rel_classes))
                            or "None")
                lines.append(
                    f"object: {name} attention: {attn_str}, "
                    f"spatial: {spat_str}, contact: {cont_str}."
                )
            text = "\n".join(lines)
            interval_texts.append(text)

            if max_tokens is not None:
                # 用 tokenizer.encode 统计 token 数（不计入特殊 tokens）
                cnt = len(tokenizer.encode(text, add_special_tokens=False))
                token_counts.append(cnt)

        # 3. 如果超过 max_tokens，构建前缀和，一次性丢弃最早区间
        if max_tokens is not None and sum(token_counts) > max_tokens:
            prefix_sums = []
            s = 0
            for cnt in token_counts:
                s += cnt
                prefix_sums.append(s)
            overflow = sum(token_counts) - max_tokens
            # 找到第一个使得前缀和 >= overflow 的索引 i
            drop_i = next(i for i, ps in enumerate(prefix_sums) if ps >= overflow)
            # 丢掉 0..drop_i，保留其后的所有 interval_texts
            interval_texts = interval_texts[drop_i + 1:]

        # 4. 拼回最终输出
        return "\n".join(interval_texts)

    def _aggregate_scene_graph_ctrl_len(self, frames, max_len = 2):
        """Aggregate scene graph across frames and merge identical consecutive frames,
        ensuring each merged interval does not exceed three frames.
        """
        intervals = []
        start_idx = 0
        n = len(frames)
        max_group_size = max_len  # Maximum number of frames per group
        
        while start_idx < n:
            end_idx = start_idx
            # Limit the group size to max_group_size
            while (end_idx + 1 < n and 
                self._compare_frame_data(frames[end_idx], frames[end_idx + 1]) and
                (end_idx + 1 - start_idx) < max_group_size):
                end_idx += 1
            # If the next frame is identical but adding it would exceed max_group_size,
            # do not include it in this group
            if (end_idx + 1 < n and 
                self._compare_frame_data(frames[end_idx], frames[end_idx + 1]) and
                (end_idx + 1 - start_idx) == max_group_size):
                pass  # Do not increment end_idx further
            frame_start = self._extract_frame_number(frames[start_idx][0].get('frame', '0'))
            frame_end = self._extract_frame_number(frames[end_idx][0].get('frame', '0'))
            intervals.append((frame_start, frame_end, frames[start_idx]))
            start_idx = end_idx + 1
        
        all_lines = []
        for (fr_s, fr_e, frame_data) in intervals:
            if fr_s == fr_e:
                all_lines.append(f"Frame {fr_s}:")
            else:
                all_lines.append(f"Frame {fr_s}-{fr_e}:")
            for obj in frame_data[1:]:
                cls_idx = obj.get('class', -1)
                obj_name = self.obj_classes[cls_idx] if 0 <= cls_idx < len(self.obj_classes) else "unknown"
                attn_ids = obj.get('attention_relationship',[])
                if hasattr(attn_ids, 'tolist'):
                    attn_ids = attn_ids.tolist()
                attn_str = ",".join([self.attn_rel_classes[i] for i in attn_ids]) if attn_ids else "None"
                spat_ids = obj.get('spatial_relationship',[])
                if hasattr(spat_ids, 'tolist'):
                    spat_ids = spat_ids.tolist()
                spat_str = ",".join([self.spat_rel_classes[i] for i in spat_ids]) if spat_ids else "None"
                cont_ids = obj.get('contacting_relationship',[])
                if hasattr(cont_ids, 'tolist'):
                    cont_ids = cont_ids.tolist()
                cont_str = ",".join([self.cont_rel_classes[i] for i in cont_ids]) if cont_ids else "None"
                line = f"object: {obj_name} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
                all_lines.append(line)
        return "\n".join(all_lines)    
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

    def _build_text_from_segments_for_object(self, obj_segments, obj_cls, observed=True, include_time=True, max_tokens=None):
        """
        原来用于生成整体 prompt 或 target 的文本（将多个段拼接成一段）
        这里调用 _construct_segment_text，不添加时间信息也不使用 <obj> 标记。
        
        参数：
        - obj_segments: 所有段落列表
        - obj_cls: 对象类别
        - observed: 是否为观察模式
        - include_time: 是否包含时间信息
        - max_tokens: 最大token数量限制，默认2100，超过此限制将截取最近的完整段落
        """
        # 如果未提供 max_tokens 或 max_tokens <= 0，保持原始逻辑
        if max_tokens is None or max_tokens <= 0:
            lines = []
            for i, seg in enumerate(obj_segments):
                if i > 0:
                    start_time = end_time
                else: 
                    start_time = seg["start_time"]
                end_time = seg["end_time"] + 1
                line = self._construct_segment_text(start_time, seg['end_time'], seg, obj_cls, include_time=include_time, add_obj_marker=True, ignore_obj_mode=False)
                lines.append(line)
            return "\n".join(lines)
        
        # 首先按时间排序（时间升序）
        sorted_segments = sorted(obj_segments, key=lambda x: x["start_time"])
        
        # 为每个段落构建文本
        segment_texts = []
        for i, seg in enumerate(sorted_segments):
            if i > 0:
                start_time = end_time
            else: 
                start_time = seg["start_time"]
            end_time = seg["end_time"] + 1
            line = self._construct_segment_text(start_time, seg['end_time'], seg, obj_cls, include_time=include_time, add_obj_marker=True, ignore_obj_mode=False)
            segment_texts.append(line)
        
        # 使用 tokenizer 计算 token 数量
        tokenizer = self.stage1_anticipator.tokenizer
        
        # 从最新的段落开始，计算并筛选在 token 限制内的段落
        total_tokens = 0
        selected_segments = []
        
        # 反向遍历段落（从最近的开始）
        for text in reversed(segment_texts):
            # 计算当前段落的 token 数量
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            
            # 检查是否超过限制
            if total_tokens + tokens + len(selected_segments) <= max_tokens:  # +len 考虑换行符
                selected_segments.insert(0, text)  # 在列表前面插入（保持时间升序）
                total_tokens += tokens + 1  # +1 为换行符
            else:
                # 如果添加此段落会超出限制，停止添加
                break
        
        # 如果没有选择任何段落（所有段落都太长），至少保留最后一个段落
        if not selected_segments and segment_texts:
            last_text = segment_texts[-1]
            selected_segments = [last_text]
        
        return "\n".join(selected_segments)
    
    def _no_merge_frames_for_objects_inference(self, frames_annotation):
        """
        不合并连续帧中相同状态的物体，为每帧的每个物体创建独立的段落记录。
        
        参数:
            frames_annotation: 包含每个视频帧标注信息的列表
            
        返回:
            segments: 每个物体在每帧的状态记录，不进行合并
        """
        segments = []
        
        # 遍历每一帧
        for idx_frame, frame_data in enumerate(frames_annotation):
            # 提取帧号
            raw_frame_str = frame_data[0].get('frame','')
            filename = raw_frame_str.split('/')[-1]
            frame_num_str = filename.replace('.png','')
            real_time = int(frame_num_str)
            
            # 获取当前帧中的所有物体
            objs = frame_data[1:]
            
            # 处理每个物体
            for obj_dict in objs:
                # 获取物体类别
                cls_idx = obj_dict.get('class', -1)
                if 0 <= cls_idx < len(self.obj_classes):
                    obj_class = self.obj_classes[cls_idx]
                else:
                    obj_class = "unknown"
                
                # 获取关系索引
                attn_ids = obj_dict.get('attention_relationship', [])
                spat_ids = obj_dict.get('spatial_relationship', [])
                cont_ids = obj_dict.get('contacting_relationship', [])
                
                # 转换为列表（如果是张量）
                if hasattr(attn_ids, 'tolist'):
                    attn_ids = attn_ids.tolist()
                if hasattr(spat_ids, 'tolist'):
                    spat_ids = spat_ids.tolist()
                if hasattr(cont_ids, 'tolist'):
                    cont_ids = cont_ids.tolist()
                
                # 转换为全局关系索引
                attn_abs = [self.rel_classes.index(self.attn_rel_classes[i]) for i in attn_ids]
                spat_abs = [self.rel_classes.index(self.spat_rel_classes[i]) for i in spat_ids]
                cont_abs = [self.rel_classes.index(self.cont_rel_classes[i]) for i in cont_ids]
                
                # 直接创建段落记录，不进行合并
                segments.append({
                    "object_class": obj_class,
                    "attn_ids": attn_abs,
                    "spat_ids": spat_abs,
                    "cont_ids": cont_abs,
                    "start_time": real_time,
                    "end_time": real_time  # 开始时间和结束时间相同，表示单帧
                })
        
        # 按开始时间排序
        segments.sort(key=lambda x: x["start_time"])
        return segments

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
        text = f"{time_text}{obj_text} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
        
        # text = f"{time_text}{obj_text} Person attention to the object: {attn_str}, the object located relative to person: {spat_str}, Person contact with the object: {cont_str}."
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
                parts = line.split("attention:")
                if len(parts) > 1:
                    after_attn = parts[1]
                    attn_split = after_attn.split("spatial:")
                    attn_part = attn_split[0].strip().rstrip(',')

                    if len(attn_split) > 1:
                        after_spat = attn_split[1]
                        spat_split = after_spat.split("contact:")
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
                # if len(attn_rels) + len(spat_rels) + len(cont_rels) > 3:
                #     print(f"Warning: Too many relations in line: {line}")

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

    def classify_generated_text_for_object(self, lines, obj_cls, device=None, parse = False):
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
            logits = torch.sigmoid(logits)
                # 如果 parse=True，则解析文本并修改 logits
            if parse:
                num_classes = len(self.rel_classes)
                for i, line in enumerate(lines):
                    try:
                        # 提取 Attention, Spatial, Contact 部分
                        parts = line.split("attention:")
                        if len(parts) > 1:
                            after_attn = parts[1]
                            attn_split = after_attn.split("spatial:")
                            attn_part = attn_split[0].strip().rstrip(',')

                            if len(attn_split) > 1:
                                after_spat = attn_split[1]
                                spat_split = after_spat.split("contact:")
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

                        boost_factor = 2.0
                        # 对每个关系查找索引并将 logits 置为较大值
                        for rel in attn_rels:
                            if rel in self.rel_classes:
                                rel_idx = self.rel_classes.index(rel)
                                logits[i, rel_idx] = 1  # 使 sigmoid(10) 接近 1
                        
                        for j, rel in enumerate(spat_rels):
                            if rel in self.rel_classes:
                                rel_idx = self.rel_classes.index(rel)
                                # logits[i, rel_idx] = 1
                                original_score = logits[i, rel_idx]
                                boosted_score = original_score * boost_factor
                                logits[i, rel_idx] = max(max(boosted_score, original_score + 0.3), 1.0)
                        
                        for j, rel in enumerate(cont_rels):
                            if rel in self.rel_classes:
                                rel_idx = self.rel_classes.index(rel)
                                # logits[i, rel_idx] = 1
                                original_score = logits[i, rel_idx]
                                boosted_score = original_score * boost_factor
                                logits[i, rel_idx] = max(max(boosted_score, original_score + 0.3), 1.0)

                    except Exception as e:
                        print(f"解析行时出错: {line}, 错误: {e}")
            # probs = torch.sigmoid(logits)
            # print(f"logits: {logits}")
            # breakpoint()
        
        return logits.cpu()

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

    def parse_stage0_output(self, generated_text, future_frames):
        """解析 Stage 0 的输出，提取每帧的物体列表"""
        pattern = re.compile(r"Frame (\d+):\s*(.*)")
        objects_by_frame = {}
        lines = generated_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            match = pattern.search(line)
            if match:
                frame_num = int(match.group(1))
                if frame_num in future_frames:
                    objs_str = match.group(2)
                    objects_raw = [obj.strip() for obj in objs_str.split(',') if obj.strip()]
                
                    # 过滤并去重对象
                    valid_objects = []
                    for obj in objects_raw:
                        # 检查对象是否在预定义类别中
                        if obj in self.obj_classes:
                            # 检查是否已经在当前帧中记录过
                            if obj not in valid_objects:
                                valid_objects.append(obj)
                    
                    objects_by_frame[frame_num] = valid_objects

        # 填充缺失的帧
        for frame in future_frames:
            if frame not in objects_by_frame:
                objects_by_frame[frame] = ["None"]
        return objects_by_frame
