from lib.supervised.config import Config
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from train_sga_base import TrainSGABase
import json
import os
import csv
import torch
from object_classes import OBJECT_CLASSES, RELATIONSHIP_CLASSES
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# -------------------------------------------------------------------------------------
# ------------------------------- BASELINE METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainSTTranAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.sttran_ant import STTranAnt
        self._model = STTranAnt(mode=self._conf.mode,
                                attention_class_num=len(self._test_dataset.attention_relationships),
                                spatial_class_num=len(self._test_dataset.spatial_relationships),
                                contact_class_num=len(self._test_dataset.contacting_relationships),
                                obj_classes=self._test_dataset.object_classes,
                                enc_layer_num=self._conf.enc_layer,
                                dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = False

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainSTTranGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.sttran_gen_ant import STTranGenAnt

        self._model = STTranGenAnt(mode=self._conf.mode,
                                   attention_class_num=len(self._test_dataset.attention_relationships),
                                   spatial_class_num=len(self._test_dataset.spatial_relationships),
                                   contact_class_num=len(self._test_dataset.contacting_relationships),
                                   obj_classes=self._test_dataset.object_classes,
                                   enc_layer_num=self._conf.enc_layer,
                                   dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = False #True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = False #True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainDsgDetrAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def init_model(self):
        from lib.supervised.sga.dsgdetr_ant import DsgDetrAnt

        self._model = DsgDetrAnt(mode=self._conf.mode,
                                 attention_class_num=len(self._test_dataset.attention_relationships),
                                 spatial_class_num=len(self._test_dataset.spatial_relationships),
                                 contact_class_num=len(self._test_dataset.contacting_relationships),
                                 obj_classes=self._test_dataset.object_classes,
                                 enc_layer_num=self._conf.enc_layer,
                                 dec_layer_num=self._conf.dec_layer).to(device=self._device)
        self._init_matcher()
        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = False

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainDsgDetrGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def init_model(self):
        from lib.supervised.sga.dsgdetr_gen_ant import DsgDetrGenAnt

        self._model = DsgDetrGenAnt(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    enc_layer_num=self._conf.enc_layer,
                                    dec_layer_num=self._conf.dec_layer).to(device=self._device)
        self._init_matcher()
        self._init_transformer_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.max_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        num_ff = self._conf.max_window
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


# -------------------------------------------------------------------------------------
# ------------------------------- SCENE SAYER METHODS ---------------------------------
# -------------------------------------------------------------------------------------
    
class TrainODE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        if self._conf.use_llm:
            from lib.supervised.sga.scene_sayer_ode_inte_llm import SceneSayerODE
        else:  
            from lib.supervised.sga.scene_sayer_ode import SceneSayerODE  
        if self._conf.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(os.environ['LOCAL_RANK'])  
            self._model = SceneSayerODE(mode=self._conf.mode,
                                        attention_class_num=len(self._test_dataset.attention_relationships),
                                        spatial_class_num=len(self._test_dataset.spatial_relationships),
                                        contact_class_num=len(self._test_dataset.contacting_relationships),
                                        obj_classes=self._test_dataset.object_classes,
                                        rel_classes=self._test_dataset.relationship_classes,
                                        max_window=self._conf.max_window,
                                        script_required=self._conf.script_require,
                                        object_required=self._conf.object_required,
                                        relation_required=self._conf.relation_required,
                                        use_classify_head=self._conf.use_classify_head,
                                        llama_path=self._conf.llama_path,
                                        lora_path=self._conf.lora_path,
                                        classifier_path=self._conf.classifier_path,
                                        use_fusion=self._conf.use_fusion,
                                        save_path=self._conf.save_path,
                                        ).cuda(local_rank)
            print(f"Using DistributedDataParallel on device {local_rank}")
            self._model = DDP(self._model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            self._model = SceneSayerODE(mode=self._conf.mode,
                                        attention_class_num=len(self._test_dataset.attention_relationships),
                                        spatial_class_num=len(self._test_dataset.spatial_relationships),
                                        contact_class_num=len(self._test_dataset.contacting_relationships),
                                        obj_classes=self._test_dataset.object_classes,
                                        rel_classes=self._test_dataset.relationship_classes,
                                        max_window=self._conf.max_window,
                                        script_required=self._conf.script_require,
                                        object_required=self._conf.object_required,
                                        relation_required=self._conf.relation_required,
                                        use_classify_head=self._conf.use_classify_head,
                                        llama_path=self._conf.llama_path,
                                        lora_path=self._conf.lora_path,
                                        classifier_path=self._conf.classifier_path,
                                        use_fusion=self._conf.use_fusion,
                                        save_path=self._conf.save_path,
                                        ).to(device=self._conf.device)

        self._init_matcher()
        self._init_diffeq_loss_function_heads()
        self.OBJECT_CLASSES = OBJECT_CLASSES
        self.RELATIONSHIP_CLASSES = RELATIONSHIP_CLASSES

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        if self._conf.use_llm:
            self._enable_ant_pred_loss = False
            self._enable_ant_bb_subject_loss = False
            self._enable_ant_bb_object_loss = False
            self._enable_ant_recon_loss = False
        else:
            self._enable_ant_pred_loss = True
            self._enable_ant_bb_subject_loss = False
            self._enable_ant_bb_object_loss = False
            self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size, script_embeddings = None, video_id = None) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        if script_embeddings is not None:
            entry["script_embeddings"] = script_embeddings
        if video_id is not None:
            entry["video_id"] = video_id
        
        pred = self._model(entry)
        # if video_id is not None:
        #     frame_indices = entry["frame_idx"]
        #     # 检查frame_indices是否有足够的帧
        #     if len(frame_indices) < 10:
        #         print(f"警告: frame_indices长度不足10，当前长度为{len(frame_indices)}，跳过该视频。")
        #         return pred  # 跳过当前视频的处理

        #     # print(f"frame_indices: {frame_indices}, count: {len(frame_indices)}")  # 打印帧索引和数量
        #     # 假设 input_gt 和 future_gt 都是包含5帧的列表
        #     input_gt_scene_graphs = []
        #     pred_scene_graphs = []
        #     future_gt_scene_graphs = []

        #     # 假设当前帧是第5帧，前5帧为输入，后5帧为未来
        #     # 需要根据实际情况调整
        #     n = len(frame_indices)
        #     input_frames = frame_indices[:n]

        #     # future_frames = frame_indices[n:]
        #     # breakpoint()

        #     for idx in range(n):
        #         frame_id = input_frames[idx]
        #         # breakpoint()
        #         gt_frame = gt_annotation[idx] if isinstance(gt_annotation, list) else gt_annotation
        #         input_scene_graph = self.extract_ground_truth_scene_graph(gt_frame)
        #         input_gt_scene_graphs.append(input_scene_graph)
        #     # 预测未来n帧
        #     # for idx in range(5):
        #     #     pred_frame = {k: v[idx].detach().cpu() if isinstance(v, torch.Tensor) and v.size(0) > idx else None for k, v in pred.items()}
        #     #     pred_scene_graph = self.extract_prediction_scene_graph(pred_frame) if pred_frame else []
        #     #     pred_scene_graphs.append(pred_scene_graph)
            
        #     # 未来5帧的Ground Truth
        #     # for idx in range(len(frame_indices) - n):
        #     #     frame_id_future = future_frames[idx]
        #     #     gt_frame_future = gt_annotation[frame_id_future] if isinstance(gt_annotation, list) else gt_annotation
        #     #     future_scene_graph = self.extract_ground_truth_scene_graph(gt_frame_future)
        #     #     future_gt_scene_graphs.append(future_scene_graph)
            
        #     # 保存结果
        #     self.save_scene_graph_results(video_id, frame_id, input_gt_scene_graphs, pred_scene_graphs = None, future_gt_scene_graphs = None)

        # if video_id is not None:
        #     frame_indices = entry["frame_idx"]
        #     print(f"frame_indices: {frame_indices}, count: {len(frame_indices)}")  # 打印帧索引和数量
        #     # 确保 pred 是一个字典，并且各个键对应的值是列表或张量
        #     for idx, frame_id in enumerate(frame_indices):
        #         # for k, v in pred.items():
        #         #     if isinstance(v, torch.Tensor):
        #         #         print(f"{k}: {v.size()}")
        #         # 提取单帧的预测结果
        #         pred_frame = {k: v[idx].detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
        #         breakpoint() 
        #         # 提取单帧的 Ground Truth
        #         gt_frame = gt_annotation[idx] if isinstance(gt_annotation, list) else gt_annotation
        #         # 提取 scene graph
        #         pred_scene_graph = self.extract_prediction_scene_graph(pred_frame)
        #         gt_scene_graph = self.extract_ground_truth_scene_graph(gt_frame)
        #         breakpoint() 
        #         self.save_scene_graph_results(video_id, frame_id, pred_scene_graph, gt_scene_graph)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size, script_embeddings = None) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size,self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        if script_embeddings is not None:
            entry["script_embeddings"] = script_embeddings
        pred = self._model(entry, True)
        
        # 获取 frame_idx 数组
        frame_indices = entry.get("frame_idx", [])
        video_id = entry.get("video_id", "unknown_video")

        # 确保 pred 是一个字典，并且各个键对应的值是列表或张量
        for idx, frame_id in enumerate(frame_indices):
            # 提取单帧的预测结果
            pred_frame = {k: v[idx].detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
            # 提取单帧的 Ground Truth
            gt_frame = gt_annotation[idx] if isinstance(gt_annotation, list) else gt_annotation
            # 提取 scene graph
            pred_scene_graph = self.extract_prediction_scene_graph(pred_frame)
            gt_scene_graph = self.extract_ground_truth_scene_graph(gt_frame)
            self.save_scene_graph_results(video_id, frame_id, pred_scene_graph, gt_scene_graph)
        return pred
    
    def save_scene_graph_results(self, video_id, frame_id, input_gt_scene_graphs = None, pred_scene_graphs = None, future_gt_scene_graphs = None):
        """
        保存场景图结果到CSV文件，分成三个部分：输入帧的GT、预测帧的Scene Graph、未来帧的GT。

        :param video_id: 视频ID
        :param frame_id: 当前帧ID
        :param input_gt_scene_graphs: 输入帧的Ground Truth列表（包含5帧）
        :param pred_scene_graphs: 预测帧的Scene Graph列表（包含5帧）
        :param future_gt_scene_graphs: 未来帧的Ground Truth列表（包含5帧）
        """
        base_save_path = self._conf.save_path

        # 定义三个文件路径
        file_paths = {
            "Input_GT": os.path.join(base_save_path, "input_gt_scene_graph_results.csv"),
            "Predictions": os.path.join(base_save_path, "pred_scene_graph_results.csv"),
            "Future_GT": os.path.join(base_save_path, "future_gt_scene_graph_results.csv")
        }

        headers = [
        "video_id", "frame_id", 
        "frame_number", "subject_idx", "object_idx",
        "attention_relation", "spatial_relation", "contacting_relation"
    ]

        # 定义三个部分及对应的数据
        sections = [
            ("Input_GT", input_gt_scene_graphs),
            ("Predictions", pred_scene_graphs),
            ("Future_GT", future_gt_scene_graphs)
        ]

        for section_name, scene_graphs in sections:
            if scene_graphs is None: 
                continue
            save_path = file_paths[section_name]
            file_exists = os.path.isfile(save_path)
            with open(save_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(headers)
                
                for frame_num, scene_graph in enumerate(scene_graphs, start=1):
                    for sg in scene_graph:
                        writer.writerow([
                            video_id, frame_id, 
                            frame_num,
                            sg.get('subject_idx', 'none'), 
                            sg.get('object_idx', 'none'),
                            sg.get('attention_relation', 'none'), 
                            sg.get('spatial_relation', 'none'), 
                            sg.get('contacting_relation', 'none')
                        ])
    def extract_prediction_scene_graph(self, pred_frame):
        """
        从预测结果中提取 scene graph 三元组 (subject, attention_relation, spatial_relation, contacting_relation, object)。
        """
        triplets = []
        required_keys = ["pred_labels", "pair_idx", "anticipated_attention_distribution", 
                        "anticipated_spatial_distribution", "anticipated_contacting_distribution"]
        
        # 检查必要的键是否存在
        if all(key in pred_frame for key in required_keys):
            pred_labels = pred_frame["pred_labels"].tolist()
            pair_idx = pred_frame["pair_idx"].tolist()
            attention_dist = pred_frame["anticipated_attention_distribution"]
            spatial_dist = pred_frame["anticipated_spatial_distribution"]
            contacting_dist = pred_frame["anticipated_contacting_distribution"]
            
            # 确保 pred_labels 和 pair_idx 是列表
            if not isinstance(pred_labels, list):
                pred_labels = [pred_labels]
            if not isinstance(pair_idx, list):
                pair_idx = [pair_idx]
            
            for label, pair in zip(pred_labels, [pair_idx]):
                # 获取关系类别，防止索引越界
                relation = self.RELATIONSHIP_CLASSES[label] if label < len(self.RELATIONSHIP_CLASSES) else "unknown"
                
                subj_idx, obj_idx = pair[0], pair[1]
                
                # 提取各类关系的预测
                if attention_dist.size(0) > subj_idx and attention_dist.size(1) > obj_idx:
                    attention_rel_idx = torch.argmax(attention_dist[subj_idx, obj_idx]).item()
                    attention_rel = self.RELATIONSHIP_CLASSES[attention_rel_idx] if attention_rel_idx < len(self.RELATIONSHIP_CLASSES) else "unknown"
                else:
                    attention_rel = "unknown"
                
                if spatial_dist.size(0) > subj_idx and spatial_dist.size(1) > obj_idx:
                    spatial_rel_idx = torch.argmax(spatial_dist[subj_idx, obj_idx]).item()
                    spatial_rel = self.RELATIONSHIP_CLASSES[spatial_rel_idx] if spatial_rel_idx < len(self.RELATIONSHIP_CLASSES) else "unknown"
                else:
                    spatial_rel = "unknown"
                
                if contacting_dist.size(0) > subj_idx and contacting_dist.size(1) > obj_idx:
                    contacting_rel_idx = torch.argmax(contacting_dist[subj_idx, obj_idx]).item()
                    contacting_rel = self.RELATIONSHIP_CLASSES[contacting_rel_idx] if contacting_rel_idx < len(self.RELATIONSHIP_CLASSES) else "unknown"
                else:
                    contacting_rel = "unknown"
                
                subject = self.OBJECT_CLASSES[subj_idx] if subj_idx < len(self.OBJECT_CLASSES) else "unknown"
                object = self.OBJECT_CLASSES[obj_idx] if obj_idx < len(self.OBJECT_CLASSES) else "unknown"
                triplet = {
                    "subject_idx": subject,
                    "attention_relation": attention_rel,
                    "spatial_relation": spatial_rel,
                    "contacting_relation": contacting_rel,
                    "object_idx": object
                }
                
                triplets.append(triplet)
        
        return triplets

    def extract_ground_truth_scene_graph(self, gt_frame):
        """
        从 Ground Truth 中提取 scene graph 三元组 (subject, attention_relation, spatial_relation, contacting_relation, object)。
        处理多个关系类型（注意、空间、接触）。
        """
        triplets = []
        subject_idx = 0  # 假设 person 是主体，索引为0

        if isinstance(gt_frame, list) and len(gt_frame) > 1:
            objects = gt_frame[1:]  # 假设第一个元素是 person_bbox
            for obj in objects:
                obj_idx = objects.index(obj) + 1  # 对象索引从1开始

                # 初始化关系为 "unknown"
                attention_rel = "unknown"
                spatial_rel = "unknown"
                contacting_rel = "unknown"

                # 处理注意关系
                if 'attention_relationship' in obj:
                    attention_rels = obj['attention_relationship']
                    if isinstance(attention_rels, torch.Tensor):
                        attention_rels = attention_rels.tolist()
                    if isinstance(attention_rels, list) and len(attention_rels) > 0:
                        rel_idx = attention_rels[0]
                        if 0 <= rel_idx < len(self.RELATIONSHIP_CLASSES):
                            attention_rel = self.RELATIONSHIP_CLASSES[rel_idx]
                        else:
                            print(f"警告：attention_relationship 索引 {rel_idx} 超出范围")
                    elif isinstance(attention_rels, int):
                        rel_idx = attention_rels
                        if 0 <= rel_idx < len(self.RELATIONSHIP_CLASSES):
                            attention_rel = self.RELATIONSHIP_CLASSES[rel_idx]
                        else:
                            print(f"警告：attention_relationship 索引 {rel_idx} 超出范围")

                # 处理空间关系
                if 'spatial_relationship' in obj:
                    spatial_rels = obj['spatial_relationship']
                    if isinstance(spatial_rels, torch.Tensor):
                        spatial_rels = spatial_rels.tolist()
                    if isinstance(spatial_rels, list) and len(spatial_rels) > 0:
                        rel_idx = spatial_rels[0]
                        if 0 <= rel_idx < len(self.RELATIONSHIP_CLASSES):
                            spatial_rel = self.RELATIONSHIP_CLASSES[rel_idx]
                        else:
                            print(f"警告：spatial_relationship 索引 {rel_idx} 超出范围")
                    elif isinstance(spatial_rels, int):
                        rel_idx = spatial_rels
                        if 0 <= rel_idx < len(self.RELATIONSHIP_CLASSES):
                            spatial_rel = self.RELATIONSHIP_CLASSES[rel_idx]
                        else:
                            print(f"警告：spatial_relationship 索引 {rel_idx} 超出范围")

                # 处理接触关系
                if 'contacting_relationship' in obj:
                    contacting_rels = obj['contacting_relationship']
                    if isinstance(contacting_rels, torch.Tensor):
                        contacting_rels = contacting_rels.tolist()
                    if isinstance(contacting_rels, list) and len(contacting_rels) > 0:
                        rel_idx = contacting_rels[0]
                        if 0 <= rel_idx < len(self.RELATIONSHIP_CLASSES):
                            contacting_rel = self.RELATIONSHIP_CLASSES[rel_idx]
                        else:
                            print(f"警告：contacting_relationship 索引 {rel_idx} 超出范围")
                    elif isinstance(contacting_rels, int):
                        rel_idx = contacting_rels
                        if 0 <= rel_idx < len(self.RELATIONSHIP_CLASSES):
                            contacting_rel = self.RELATIONSHIP_CLASSES[rel_idx]
                        else:
                            print(f"警告：contacting_relationship 索引 {rel_idx} 超出范围")

                subject = self.OBJECT_CLASSES[subject_idx] if subject_idx < len(self.OBJECT_CLASSES) else "unknown"
                object = self.OBJECT_CLASSES[obj_idx] if obj_idx < len(self.OBJECT_CLASSES) else "unknown"
                triplet = {
                    "subject_idx": subject,
                    "attention_relation": attention_rel,
                    "spatial_relation": spatial_rel,
                    "contacting_relation": contacting_rel,
                    "object_idx": object
                }

                triplets.append(triplet)

        return triplets


    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.ode_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_scene_sayer_evaluation_score(pred, gt)


class TrainSDE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.scene_sayer_sde import SceneSayerSDE
        print("SceneSayerSDE")
        self._model = SceneSayerSDE(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    max_window=self._conf.max_window,
                                    brownian_size=self._conf.brownian_size).to(device=self._device)

        self._init_matcher()
        self._init_diffeq_loss_function_heads()

    def init_method_loss_type_params(self):
        # Observed Representations Loss
        self._enable_obj_class_loss = True
        self._enable_gen_pred_class_loss = True

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = True
        self._enable_ant_bb_subject_loss = True
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = True

    def process_train_video(self, entry, gt_annotation, frame_size, script_embeddings = None) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry)

        pred["script_embeddings"] = script_embeddings
        return pred

    def process_test_video_old(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        pred = self._model(entry, True)
        return pred
    
    def process_test_video(self, entry, gt_annotation, frame_size, script_embeddings = None) -> dict:
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        entry["gt_annotation"] = gt_annotation
        if script_embeddings is not None:
            entry["script_embeddings"] = script_embeddings
        pred = self._model(entry, True)
        
        # 获取 frame_idx 数组
        frame_indices = entry.get("frame_idx", [])
        video_id = entry.get("video_id", "unknown_video")

        # 处理每一帧
        for idx, frame_id in enumerate(frame_indices):
            # 提取单帧的预测结果，修复0维张量索引错误
            pred_frame = {}
            for k, v in pred.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0:  # 0维张量(标量)
                        pred_frame[k] = v.item()  # 使用item()
                    elif idx < v.shape[0]:  # 确保索引在范围内
                        pred_frame[k] = v[idx].detach().cpu()
                    else:
                        # 索引超出范围或其他情况，保留原样
                        pred_frame[k] = v.detach().cpu()
                else:
                    pred_frame[k] = v
                    
            # 提取单帧的Ground Truth并处理scene graph
            gt_frame = gt_annotation[idx] if isinstance(gt_annotation, list) else gt_annotation
            pred_scene_graph = self.extract_prediction_scene_graph(pred_frame)
            gt_scene_graph = self.extract_ground_truth_scene_graph(gt_frame)
            self.save_scene_graph_results(video_id, frame_id, pred_scene_graph, gt_scene_graph)
        
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.sde_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_scene_sayer_evaluation_score(pred, gt)


# -------------------------------------------------------------------------------------
# def init_distributed_mode():
#     import torch.distributed as dist
#     if 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ['RANK'])
#         world_size = int(os.environ['WORLD_SIZE'])
#         local_rank = int(os.environ['LOCAL_RANK'])
#         dist.init_process_group(backend='nccl', init_method='env://')
#         torch.cuda.set_device(local_rank)
#         print(f'Using distributed mode on device {local_rank}')
#         return local_rank
#     else:
#         print('Not using distributed mode')
#         return 0
    
def main():
    conf = Config()
    if conf.distributed:
        import torch.distributed as dist
        import torch.multiprocessing as mp
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        world_size = dist.get_world_size()
        conf.local_rank = local_rank
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
    conf.device = device
    if conf.method_name == "ode":
        evaluate_class = TrainODE(conf)
    elif conf.method_name == "sde":
        evaluate_class = TrainSDE(conf)
    elif conf.method_name == "sttran_ant":
        evaluate_class = TrainSTTranAnt(conf)
    elif conf.method_name == "sttran_gen_ant":
        evaluate_class = TrainSTTranGenAnt(conf)
    elif conf.method_name == "dsgdetr_ant":
        evaluate_class = TrainDsgDetrAnt(conf)
    elif conf.method_name == "dsgdetr_gen_ant":
        evaluate_class = TrainDsgDetrGenAnt(conf)
    else:
        raise NotImplementedError

    evaluate_class.init_method_training()


if __name__ == "__main__":
    main()
