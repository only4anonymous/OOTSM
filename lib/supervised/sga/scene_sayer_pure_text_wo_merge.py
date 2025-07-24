import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import time
import copy
import numpy as np
import requests  # 新增：用于调用外部API
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint_adjoint as odeint

# 以下 import 根据你项目路径修改
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 你已有的 STTran 实现
from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierTransformer, GetBoxes
from lib.word_vectors import obj_edge_vectors

# 下列 import 保持不变（部分用于 stage0 和 stage2，如有需要可以保留）
from llama_SGA.SGA_stage_1 import SceneGraphFineTuner
from llama_SGA.SGA_stage_2 import SceneGraphAllocator
from llama_SGA.SGA_stage_0 import SceneGraphAllocator as SceneGraphFineTuner0
import openai
from openai import AzureOpenAI, OpenAI

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

_PATTERN_LINE = re.compile(
    r'.*?attention:\s*([^,]*?(?:,[^,]*?)*?)(?=,\s*spatial:),\s*spatial:\s*([^,]*?(?:,[^,]*?)*?)(?=,\s*contact:),\s*contact:\s*([^,]*?(?:,[^,]*?)*?)(?:\.|\s|$)',
    re.IGNORECASE
)

def call_text_generation_hkust_api(prompt, model = "gpt-4o-mini"):
    """
    使用 Azure OpenAI API 调用文本生成
    """
    
    relationship_categories = {
            "Attention": ATTN_REL_CLASSES,
            "Spatial": SPAT_REL_CLASSES,
            "Contact": CONT_REL_CLASSES
        }
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
    # 构建 messages 格式的 prompt
    messages = [
        {"role": "system", "content": header},
        {"role": "user", "content": prompt}
    ]
    if model.startswith("deepseek"):
        client = OpenAI(api_key="sk-473e62a005ad4c3cbd4a0824f9e3391b", base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
                    model=model,  # engine = "deployment_name"
                    messages=messages,
                    stream=False
                        )
    else:
        client = AzureOpenAI(
        api_key="cc6f9f7b38f94488afd39e4d60c1921d",  # your api key
        api_version="2023-05-15",            # change to newer API version
        azure_endpoint="https://hkust.azure-api.net"
        )
        response = client.chat.completions.create(
                    model=model,  # engine = "deployment_name"
                    messages=messages,
                    max_tokens=512,
                        )
    return response.choices[0].message.content


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
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class  = entry['pred_labels'][entry['pair_idx'][:, 1]]
        attn_rel_indices = torch.argmax(entry["attention_distribution"], dim=1) 
        if threshold is not None:
            spatial_probs = torch.sigmoid(entry["spatial_distribution"])
            spaitial_rel_indices = []
            for i in range(spatial_probs.size(0)):
                indices = torch.where(spatial_probs[i] > threshold)[0]
                if len(indices) == 0:
                    indices = torch.topk(spatial_probs[i], 1)[1]
                elif len(indices) > 2:
                    values = spatial_probs[i][indices]
                    _, top_idx = torch.topk(values, 2)
                    indices = indices[top_idx]
                spaitial_rel_indices.append(indices)
            
            contact_probs = torch.sigmoid(entry["contacting_distribution"])
            contacting_rel_indices = []
            for i in range(contact_probs.size(0)):
                indices = torch.where(contact_probs[i] > threshold)[0]
                if len(indices) == 0:
                    indices = torch.topk(contact_probs[i], 1)[1]
                elif len(indices) > 2:
                    values = contact_probs[i][indices]
                    _, top_idx = torch.topk(values, 2)
                    indices = indices[top_idx]
                contacting_rel_indices.append(indices)
        else:
            spaitial_rel_indices = torch.argmax(entry["spatial_distribution"], dim=1)
            contacting_rel_indices = torch.argmax(entry["contacting_distribution"], dim=1)

        return subj_class, obj_class, attn_rel_indices, spaitial_rel_indices, contacting_rel_indices

    def forward(self, entry, testing=False):
        entry = self.object_classifier(entry)
        if self.script_required and "script_embeddings" in entry and entry["script_embeddings"] is not None:
            script_emb = entry["script_embeddings"]
            script_emb = script_emb.unsqueeze(0)
            script_proj = self.script_proj(script_emb)
        else:
            script_proj = None

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
        entry["subject_boxes_dsg"] = self.get_subj_boxes(global_output)
        pair_idx = entry["pair_idx"]
        boxes_rcnn = entry["boxes"]
        entry["global_output"] = global_output
        entry["subject_boxes_rcnn"] = boxes_rcnn[pair_idx[:, 0], 1:].to(global_output.device)
        return entry

#################################
# SceneSayerODE - 替换 ODE为纯文本 API 调用
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
                 model_engine = "gpt-4o-mini"):
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
        self.max_window = max_window
        self.model_engine = model_engine

        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
        # 初始化 STTran 模块（用于生成 scene graph 注释）
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

        self.stage1_anticipator = SceneGraphFineTuner(
            model_path=llama_path,
            ckpt_path=lora_path_stage1,
            phase="eval",
            local_rank=0,  # 添加默认值
            world_size=1,  # 添加默认值
            object_classes = obj_classes,
        )

        self.relationship_categories = {
            "Attention": ATTN_REL_CLASSES,
            "Spatial": SPAT_REL_CLASSES,
            "Contact": CONT_REL_CLASSES
        }

        # 原有 Stage 0 / Stage1 / Stage2 块已移除，本次仅使用 pure text 模块进行预测。
        # 本示例将调用外部 API 来生成未来 scene graph 的纯文本描述
        # 请确保你已经配置好 API endpoint 和对应访问密钥

    # 新增：调用文本生成 API 的函数（请替换 URL 和 API_KEY）

    # 修改后的 forward_single_entry，使用纯文本 API 进行生成
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

        # Step 3: 提取 Scene 信息
        observed_anno = gt_annotation[:end + 1]
        future_anno = gt_annotation[end + 1:]
        num_future = num_frames - end - 1

        # 提取未来帧的帧号
        future_frames = [self._extract_frame_number(frame[0]['frame']) for frame in future_anno]

        # 提取最后一帧的对象列表（保持顺序并去重）
        end_frame_objects = []
        if len(observed_anno) > 0:
            last_frame = observed_anno[-1]
            for obj in last_frame[1:]:
                if 'class' in obj and 0 <= obj['class'] < len(self.obj_classes):
                    o_cls = self.obj_classes[obj['class']]
                    if o_cls not in end_frame_objects:
                        end_frame_objects.append(o_cls)

        if not end_frame_objects:
            return num_frames, {}

        # Step 4: 合并观测帧并生成文本描述
        observed_text = self._aggregate_frames_annotation(observed_anno)

        # Step 5: 构建单一提示，预测所有未来帧
        # prompt = self.build_prompt_for_multiple_objects(
        #     observed_text=observed_text,
        #     future_frames=future_frames,
        #     objects=end_frame_objects,
        #     num_future_frames=num_future
        # )
        prompt = self.stage1_anticipator.build_prompt_for_scene_graph(
                    observed_segments=observed_text,
                    object_class=obj_cls,
                    relationship_categories=self.relationship_categories,
                    future_frames=obj_to_future_frames[obj_cls]
                )

        # Step 6: 调用 API 生成文本
        # Step 6: 调用 API 生成文本
        max_attempts = 3
        overall_attempts = 0  # 整体重试计数器
        max_overall_attempts = 3  # 最大整体重试次数
        valid_generation = False

        while not valid_generation and overall_attempts < max_overall_attempts:
            overall_attempts += 1
            attempt = 0
            generated = None
            
            # 尝试生成文本
            while attempt < max_attempts and generated is None:
                try:
                    # generated = call_text_generation_hkust_api(prompt, model=self.model_engine)
                    generated_all = self.stage1_anticipator.generate_text(
                    prompts=prompt,
                    max_new_tokens=self.stage1_tokens,
                    temperature=0.7,
                    top_p=0.4
                )
                    generated = generated_all.replace(prompt, "")
                    # 初步检查生成结果是否为空或无效
                    if not generated or not any(line.startswith("Frame ") for line in generated.split("\n")):
                        generated = None
                        raise ValueError("Generated text is empty or lacks 'Frame ' sections")
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    attempt += 1
            
            if generated is None:
                print(f"警告：第 {overall_attempts} 轮尝试后，无法生成有效结果。")
                if overall_attempts >= max_overall_attempts:
                    print("达到最大重试次数，返回空结果。")
                    return num_frames, {}
                continue  # 重新尝试整个生成过程

            # Step 7: 解析生成的文本，构建 lines_batch
            lines_batch = self.parse_generated_text(generated, future_frames, end_frame_objects)

            # 检查 lines_batch 是否包含足够的行
            expected_lines = len(end_frame_objects) * num_future
            if len(lines_batch) < expected_lines:
                print(f"警告：第 {overall_attempts} 轮生成的lines_batch行数 {len(lines_batch)} 小于预期 {expected_lines}，重新尝试。")
                # 如果结果不足，继续下一轮尝试
                continue
            
            # 如果到这里，说明生成了有效的结果
            valid_generation = True

        # 如果最终没有生成有效结果，返回空结果
        if not valid_generation:
            print("警告：经过多次尝试后，仍无法生成完整预测结果。")
            return num_frames, {}

        # Step 8: 对生成的文本进行分类处理，得到关系分布矩阵
        if self.use_classify_head:
            dist_mat_all = self.classify_generated_text_for_object(lines_batch, None, device=device, parse=True)
        else:
            dist_mat_all = self.classify_generated_text_for_object_wo_classification_head(lines_batch, None)

        # Step 9: 构建 distribution_dict 和 frame_to_obj_idx
        obj_list = end_frame_objects  # 用于后续映射
        M = len(end_frame_objects)
        N = num_future
        obj_line_ranges = [(i * N, (i + 1) * N) for i in range(M)]
        distribution_dict = {}
        frame_to_obj_idx = defaultdict(list)
        obj_to_future_frames = {obj: future_frames for obj in end_frame_objects}  # 所有对象在所有未来帧中

        for i_obj, obj_cls in enumerate(obj_list):
            start_idx, end_idx = obj_line_ranges[i_obj]
            dist_mat_obj = dist_mat_all[start_idx:end_idx, :]
            obj_future_frames = future_frames
            distribution_dict[obj_cls] = {
                "dist": dist_mat_obj,
                "frames": torch.tensor(obj_future_frames, device=device)
            }
            for frame_idx in obj_future_frames:
                frame_to_obj_idx[frame_idx].append(i_obj)

        # Step 10: 生成关系分布和最终预测
        attn_mat, spat_mat, cont_mat = self.generate_relationship_distributions(
            distribution_dict, future_frames, obj_list, frame_to_obj_idx
        )
        im_idx_list = [len(frame_to_obj_idx[frame]) for frame in future_frames]
        labels_list = [[self.obj_classes.index(obj_list[i]) for i in frame_to_obj_idx[frame]] for frame in future_frames]
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
    # 以下保留原始辅助函数（如 build_frames_annotation、extract_future_segments、_extract_frame_number、等），
    # 请确保这些函数在你的代码库中均可调用
    def build_frames_annotation(
        self,
        im_idx: torch.Tensor,
        obj_class: torch.Tensor,
        attn_rel_indices: torch.Tensor,
        spatial_rel_indices,
        contacting_rel_indices,
        time: list
    ) -> list:
        assert im_idx.dim() == 1, "im_idx 必须是一维张量"
        assert obj_class.dim() == 1, "obj_class 必须是一维张量"
        im_idx = im_idx.tolist()
        obj_class = obj_class.tolist()
        if isinstance(attn_rel_indices, torch.Tensor):
            attn_rel_indices = attn_rel_indices.tolist()
        frames_dict = defaultdict(list)
        for i in range(len(im_idx)):
            frame_id = time[im_idx[i]]
            if isinstance(spatial_rel_indices, list):
                if isinstance(spatial_rel_indices[i], torch.Tensor):
                    spat_rel = spatial_rel_indices[i].tolist()
                else:
                    spat_rel = spatial_rel_indices[i]
            else:
                spat_rel = [spatial_rel_indices[i].item() if isinstance(spatial_rel_indices, torch.Tensor) 
                        else spatial_rel_indices[i]]
            if isinstance(contacting_rel_indices, list):
                if isinstance(contacting_rel_indices[i], torch.Tensor):
                    cont_rel = contacting_rel_indices[i].tolist()
                else:
                    cont_rel = contacting_rel_indices[i]
            else:
                cont_rel = [contacting_rel_indices[i].item() if isinstance(contacting_rel_indices, torch.Tensor) 
                        else contacting_rel_indices[i]]
            obj_dict = {
                'class': obj_class[i],
                'attention_relationship': [attn_rel_indices[i]],
                'spatial_relationship': spat_rel,
                'contacting_relationship': cont_rel
            }
            frames_dict[frame_id].append(obj_dict)
        frames_annotation = []
        sorted_frame_ids = sorted(frames_dict.keys())
        for frame_id in sorted_frame_ids:
            frame_meta = {'frame': f'path/to/{frame_id}.png'}
            frame_objs = frames_dict[frame_id]
            frame_entry = [frame_meta] + frame_objs
            frames_annotation.append(frame_entry)
        return frames_annotation

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

    def _extract_frame_number(self, frame_info):
        try:
            return int(frame_info.split('/')[-1].split('.')[0])
        except:
            return 0

    def _compare_frame_data(self, frame_a, frame_b):
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

    def _aggregate_frames_annotation(self, frames_annotation, max_token=None):
        # 第一步：合并连续相同帧成区间
        intervals = []
        start_idx = 0
        n = len(frames_annotation)
        while start_idx < n:
            end_idx = start_idx
            while end_idx + 1 < n and self._compare_frame_data(frames_annotation[end_idx], frames_annotation[end_idx + 1]):
                end_idx += 1
            frame_start = self._extract_frame_number(frames_annotation[start_idx][0].get('frame', '0'))
            frame_end = self._extract_frame_number(frames_annotation[end_idx][0].get('frame', '0'))
            intervals.append((frame_start, frame_end, frames_annotation[start_idx]))
            start_idx = end_idx + 1
        
        # 第二步：为每个区间生成文本并计算词数
        interval_texts = []
        for (fr_s, fr_e, frame_data) in intervals:
            lines = []
            if fr_s == fr_e:
                lines.append(f"Frame {fr_s}:")
            else:
                lines.append(f"Frame {fr_s}-{fr_e}:")
            for obj in frame_data[1:]:
                cls_idx = obj.get('class', -1)
                obj_name = self.obj_classes[cls_idx] if 0 <= cls_idx < len(self.obj_classes) else "unknown"
                
                # 处理注意力关系
                attn_ids = obj.get('attention_relationship', [])
                if len(attn_ids) > 0:
                    attn_str = ",".join([self.attn_rel_classes[i] for i in attn_ids if 0 <= i < len(self.attn_rel_classes)])
                else:
                    attn_str = "None"
                
                # 处理空间关系
                spat_ids = obj.get('spatial_relationship', [])
                if len(spat_ids) > 0:
                    spat_str = ",".join([self.spat_rel_classes[i] for i in spat_ids if 0 <= i < len(self.spat_rel_classes)])
                else:
                    spat_str = "None"
                
                # 处理接触关系
                cont_ids = obj.get('contacting_relationship', [])
                if len(cont_ids) > 0:
                    cont_str = ",".join([self.cont_rel_classes[i] for i in cont_ids if 0 <= i < len(self.cont_rel_classes)])
                else:
                    cont_str = "None"
                
                line = f"object: {obj_name} attention: {attn_str}, spatial: {spat_str}, contact: {cont_str}."
                lines.append(line)
            interval_text = "\n".join(lines)
            token_count = len(self.tokenizer.tokenize(interval_text))  # 使用 tokenizer 计算 token 数
            interval_texts.append((interval_text, token_count))
        
        # 第三步：根据 max_token 选择区间
        if max_token is None:
            # 如果未指定 max_token，返回所有区间的文本
            selected_texts = [text for text, _ in interval_texts]
        else:
            # 计算总词数
            total_words = sum(wc for _, wc in interval_texts)
            selected_texts = [text for text, _ in interval_texts]
            
            # 如果超过 max_token，从前面开始删除区间
            while total_words > max_token and selected_texts:
                # 移除最早的区间
                removed_text, removed_wc = interval_texts.pop(0)
                selected_texts.pop(0)
                total_words -= removed_wc
        
        # 第四步：生成最终文本
        observed_text = "\n".join(selected_texts)
        return observed_text

    def _group_segments_by_object_inference(self, segments):
        obj_dict = defaultdict(list)
        for seg in segments:
            obj_class = seg["object_class"]
            obj_dict[obj_class].append(seg)
        for oc in obj_dict:
            obj_dict[oc].sort(key=lambda x: x["start_time"])
        return dict(obj_dict)

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

    def classify_generated_text_for_object_wo_classification_head(self, lines, obj_cls):
        num_classes = len(self.rel_classes)
        if not lines:
            return torch.empty((0, num_classes), dtype=torch.float32)
        result = []
        for line in lines:
            row = np.zeros(num_classes, dtype=np.float32)
            try:
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
                attn_rels = [rel.strip() for rel in attn_part.split(',') if rel.strip()] if attn_part else []
                spat_rels = [rel.strip() for rel in spat_part.split(',') if rel.strip()] if spat_part else []
                cont_rels = [rel.strip() for rel in cont_part.split(',') if rel.strip()] if cont_part else []
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
            result.append(row)
        return torch.tensor(result, dtype=torch.float32)

    def classify_generated_text_for_object(self, lines, obj_cls, device=None, parse=False):
        if not lines:
            return torch.empty((0, self.attention_class_num + self.spatial_class_num + self.contact_class_num), dtype=torch.float32)
        enc = self.stage1_anticipator.tokenizer(
            lines,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device) if hasattr(self, 'stage1_anticipator') else None
        # 如果不存在内部 LLM 模块，则直接调用平均池化后的分类头（此处仍可自定义实现）
        # 这里简单调用 _wo_classification_head 方法
        return self.classify_generated_text_for_object_wo_classification_head(lines, obj_cls)

    def generate_relationship_distributions(self, distribution_dict, future_frames, obj_list, frame_to_obj_idx):
        attn_list = []
        spat_list = []
        cont_list = []
        for frame_idx in future_frames:
            obj_indices = frame_to_obj_idx[frame_idx]
            for i_obj in obj_indices:
                obj_cls = obj_list[i_obj]
                dist_info = distribution_dict[obj_cls]
                frames = dist_info["frames"]
                dist_mat = dist_info["dist"]
                mask = (frames == frame_idx)
                if mask.any():
                    k = mask.nonzero(as_tuple=True)[0][0].item()
                    row_26d = dist_mat[k]
                else:
                    raise ValueError(f"Object {obj_cls} listed in frame {frame_idx} but not found in its frames tensor.")
                attn_vec = row_26d[:self.attention_class_num]
                spat_vec = row_26d[self.attention_class_num:self.attention_class_num + self.spatial_class_num]
                cont_vec = row_26d[self.attention_class_num + self.spatial_class_num:]
                attn_list.append(attn_vec)
                spat_list.append(spat_vec)
                cont_list.append(cont_vec)
        attn_tensor = torch.stack(attn_list, dim=0) if attn_list else torch.empty((0, self.attention_class_num))
        spat_tensor = torch.stack(spat_list, dim=0) if spat_list else torch.empty((0, self.spatial_class_num))
        cont_tensor = torch.stack(cont_list, dim=0) if cont_list else torch.empty((0, self.contact_class_num))
        return attn_tensor, spat_tensor, cont_tensor

    def build_pred_from_future_frames(self, num_future_frames, im_idx_list, labels_list,
                                    attn_mat, spat_mat, cont_mat, device):
        global_labels = []
        global_scores = []
        global_pair_idx = []
        global_im_idx = []
        current_idx = 0
        for f in range(num_future_frames):
            num_objs = im_idx_list[f]
            frame_labels = [1] + labels_list[f]
            global_labels.extend(frame_labels)
            frame_scores = [1.0] * (num_objs + 1)
            global_scores.extend(frame_scores)
            global_im_idx.extend([f] * num_objs)
            for i in range(1, num_objs + 1):
                global_pair_idx.append([current_idx, current_idx + i])
            current_idx += (num_objs + 1)
        num_nodes = num_future_frames + sum(im_idx_list)
        boxes = torch.ones((num_nodes, 5), device=device) * 0.5
        pred = {}
        pred["labels"] = torch.tensor(global_labels, dtype=torch.long, device=device)
        pred["scores"] = torch.tensor(global_scores, dtype=torch.float32, device=device)
        pred["im_idx"] = torch.tensor(global_im_idx, dtype=torch.int32, device=device)
        pred["pair_idx"] = torch.tensor(global_pair_idx, dtype=torch.long, device=device)
        pred["boxes"] = boxes
        total_objects = sum(im_idx_list)
        if total_objects > 0:
            pred["attention_distribution"] = attn_mat.to(device)
            pred["spatial_distribution"] = spat_mat.to(device)
            pred["contacting_distribution"] = cont_mat.to(device)
        else:
            pred["attention_distribution"] = torch.empty((0, 3), device=device)
            pred["spatial_distribution"] = torch.empty((0, 6), device=device)
            pred["contacting_distribution"] = torch.empty((0, 17), device=device)
        return pred
    
    def _build_observed_text_from_segments(self, segments, include_time=False):
        lines = []
        for seg in segments:
            obj_cls = seg["object_class"]
            line = self._build_text_from_segments_for_object([seg], obj_cls, observed=True, include_time=include_time)
            lines.append(line)
        return "\n".join(lines) + "\n"
    
    def _build_text_from_segments_for_object(self, obj_segments, obj_cls, observed=True, include_time=True):
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
        return text
        
    def check_multiple_relations_in_text(self, text_line):
        """
        检查文本行中是否存在多个标签的关系类别，如果有则打印出来
        
        参数：
        text_line (str): 要分析的文本行
        """
        Flag = False
        try:
            # 提取 attention, spatial, contact 部分
            parts = text_line.split("attention:")
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
                
            # 提取各关系类别的标签列表
            attn_rels = [rel.strip() for rel in attn_part.split(',') if rel.strip()] if attn_part else []
            spat_rels = [rel.strip() for rel in spat_part.split(',') if rel.strip()] if spat_part else []
            cont_rels = [rel.strip() for rel in cont_part.split(',') if rel.strip()] if cont_part else []
            
            # 检查是否存在多个标签的关系类别
            if len(attn_rels) + len(spat_rels) + len(cont_rels) > 3:
                Flag = True
                # print(f"检测到多个关系类别：{attn_rels}, {spat_rels}, {cont_rels}")
                
        except Exception as e:
            print(f"检查多重关系时出错: {e}")
        return Flag
    
    def build_prompt_for_scene_graph(self, observed_segments, object_class, relationship_categories, num_future_frames, future_frames):
        
        # instruction_0 = (
        #     f"\nPlease generate the future segment for object [{object_class}] "
        #     "in the same structured format as above. "
        #     "Do not add extra commentary; output exactly in the given style.\n"
        # )
        instruction_0 = (
        f"Please generate the scene graph for object [{object_class}] in each of the following future frames: {', '.join(map(str, future_frames))}.\n"
        "Output one scene graph per frame in the following format:\n"
        "Frame {frame number}: object: {object_class} attention:..., spatial:..., contact:...\n"
        "Ensure each frame is on a separate line and no additional commentary is included.\n"
        )
        observed_text = f"Observed segment for object [{object_class}]:\n" + "\n".join(observed_segments) + "\n"
        # instruction = f"Future {num_future_frames} segments for object [{object_class}]:\n"
        future_frames_text = "Future frames " + ", ".join(map(str, future_frames)) + f" for object [{object_class}]:"+ "\n"
        prompt =  instruction_0 + observed_text + future_frames_text
        # prompt = header + instruction + observed_text + instruction
        return prompt
    
    
    def _split_generated_to_lines(self, generated_text):
        lines = generated_text.split("\n")
        lines = [ln.strip() for ln in lines if ln.strip()]
        return lines

    def parse_generated_text(self, generated_text, future_frames, end_frame_objects):
        """
        解析 LLM 生成的文本，构建 lines_batch。
        - generated_text: LLM 生成的文本，包含未来帧的预测。
        - future_frames: 未来帧的帧号列表。
        - end_frame_objects: 最后一帧的对象列表，保持顺序。
        返回 lines_batch，格式为 [obj1_frame1, obj1_frame2, ..., obj1_frameN, obj2_frame1, ...]。
        """
        # 分割成行并提取帧块
        lines = generated_text.split("\n")
        
        # 预处理：清理空行和多余空格
        lines = [line.strip() for line in lines if line.strip()]
        
        # 修复同一帧多对象的情况：将所有行按帧分组
        frame_to_objects = {}
        current_frame = None
        
        for line in lines:
            if line.startswith("Frame "):
                # 提取帧号
                try:
                    frame_num_str = line.split(":")[0].split(" ")[1]
                    current_frame = int(frame_num_str)
                    if current_frame not in frame_to_objects:
                        frame_to_objects[current_frame] = []
                    
                    # 检查这一行是否本身就包含对象信息
                    if "object:" in line:
                        obj_info = line.split("object:", 1)[1].strip()
                        if obj_info:
                            frame_to_objects[current_frame].append(f"object: {obj_info}")
                except:
                    current_frame = None
            elif current_frame is not None and "object:" in line:
                # 这是同一帧中的对象
                frame_to_objects[current_frame].append(line)
        
        # 为每个对象和每一帧构建输出
        lines_batch = []
        for obj in end_frame_objects:
            obj_lines = []
            last_line = None
            
            for frame_num in future_frames:
                if frame_num in frame_to_objects:
                    found = False
                    for obj_line in frame_to_objects[frame_num]:
                        # 提取对象名称
                        parts = obj_line.split(" attention:", 1)
                        if len(parts) == 2:
                            obj_part = parts[0][len("object:"):].strip()
                            if obj_part == obj:
                                full_line = f"Frame {frame_num}: {obj_line}" if not obj_line.startswith("Frame") else obj_line
                                obj_lines.append(full_line if "Frame" in full_line else obj_line)
                                last_line = obj_line
                                found = True
                                break
                    
                    if not found and last_line:
                        # 如果在当前帧中找不到该对象，使用上一帧的信息
                        obj_lines.append(last_line)
                elif last_line:
                    # 如果该帧完全缺失，使用上一帧信息
                    obj_lines.append(last_line)
            
            # 只有当我们找到了该对象的至少一行时，才添加到结果中
            if obj_lines:
                lines_batch.extend(obj_lines)
        
        # 输出一下解析结果，方便调试
        # print(f"解析后的场景图：{lines_batch}")
        return lines_batch

    def build_prompt_for_multiple_objects(self, observed_text, future_frames, objects, num_future_frames):
        """
        构建提示以预测所有对象的未来帧场景图，基于原始的 build_prompt_for_scene_graph 函数。

        参数：
        - observed_text (str): 过去帧的合并文本描述。
        - future_frames (list): 未来帧的帧号列表。
        - objects (list): 需要预测的对象列表。
        - num_future_frames (int): 未来帧的数量。

        返回：
        - prompt (str): 构建好的提示字符串。
        """
        objects_str = ", ".join(objects)
        frames_str = ", ".join(map(str, future_frames))
        instruction = (
            f"Please generate the scene graph for the following objects [{objects_str}] "
            f"in each of the following future frames: {frames_str}.\n"
            "Output one scene graph per frame in the following format:\n"
            "Frame {frame number}: object: {object_name} attention:..., spatial:..., contact:...\n"
            "Ensure each frame is on a separate line, include all listed objects in each frame, "
            "and provide no additional commentary.\n"
        )
        observed_segment_text = f"Observed segments for all objects:\n{observed_text}\n"
        future_frames_text = f"Future {num_future_frames} frames for objects [{objects_str}]:\n"
        prompt = instruction + observed_segment_text + future_frames_text
        return prompt