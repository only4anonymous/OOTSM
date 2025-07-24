import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from constants import Constants as const
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
import pandas as pd  # 添加此行
#导入csv的包
import csv

class AG(Dataset):

    def __init__(
            self,
            phase,
            datasize,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False,
            script_require=False,  # 新增参数
            video_id_required=False,
            relevance = False,
            verify = False, #只使用script被验证过的video
            action_classes_path=None,
            subject_id = False
    ):

        root_path = data_path
        self._phase = phase
        self._datasize = datasize
        self._data_path = data_path
        self._frames_path = os.path.join(root_path, const.FRAMES)
        self.script_require = script_require  # 设置属性
        self.video_id_required = video_id_required
        self.subject_id = subject_id

        # collect the object classes
        self._fetch_object_classes()

        # collect relationship classes
        self._fetch_relationship_classes()

        # Fetch object and person bounding boxes
        person_bbox, object_bbox = self._fetch_object_person_bboxes(self._datasize, filter_small_box)

        # collect valid frames
        video_dict, q = self._fetch_valid_frames(person_bbox, object_bbox)
        all_video_names = np.unique(q)

        # Build dataset
        self._build_dataset(video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame)
        # 如果需要脚本，加载Charades注释文件并构建映射
        if self.script_require:
            self.video_scripts = {}
            self._load_charades_scripts(verify, relevance)
        if action_classes_path is not None:
            self.action_classes = self._load_action_classes(action_classes_path)
            self.video_actions = self._load_video_actions()
        if self.subject_id:
            self.video_subjects = {}
            self._load_charades_subjects()
        

    def _load_action_classes(self, action_classes_path):
        action_classes = []
        with open(action_classes_path, 'r') as f:
            for line in f:
                action_classes.append(line.strip().split(' ', 1)[1])
        return action_classes

    def _load_video_actions(self):
        video_actions = {}
        csv_path = os.path.join(self._data_path, "annotations", f"Charades_v1_{self._phase}.csv")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['id']
                actions = row['actions'].split(';')
                video_actions[video_id] = [
                    (a.split()[0], float(a.split()[1]), float(a.split()[2]))
                    for a in actions if a
                ]
        return video_actions

    def get_frame_actions(self, video_id, frame_index):
        if video_id not in self.video_actions:
            return []
        frame_time = frame_index / 24.0  # 假设帧率为24fps
        return [
            self.action_classes[int(action[0][1:])-1]
            for action in self.video_actions[video_id]
            if action[1] <= frame_time < action[2]
        ]

    def _load_charades_scripts(self, verify = False, relevance = False):
        """
        加载Charades注释文件并构建视频ID到脚本的映射。
        """
        csv_filename = f"Charades_v1_{self._phase}.csv"
        csv_path = os.path.join(self._data_path, "annotations", csv_filename)

        if not os.path.exists(csv_path):
            print(f"警告：注释文件 {csv_path} 不存在。无法加载脚本内容。")
            return

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)

            # 应用过滤条件
            if verify:
                df = df[df['verified'] == "Yes"]
                print(f"已过滤得到 {len(df)} 个经过验证的脚本。")
                
            if relevance:
                df = df[df['relevance'] > 5]
                print(f"已过滤得到 {len(df)} 个相关性为7的脚本。")

            # 构建映射字典
            self.video_scripts = {row['id']: row['script'] for index, row in df.iterrows()}

            print(f"成功加载 {len(self.video_scripts)} 个视频的脚本内容。")
        except Exception as e:
            print(f"错误：读取CSV文件时发生异常：{e}")
    
    def _load_charades_subjects(self):
        """
        加载Charades注释文件并构建视频ID到脚本的映射。
        """
        csv_filename = f"Charades_v1_{self._phase}.csv"
        csv_path = os.path.join(self._data_path, "annotations", csv_filename)

        if not os.path.exists(csv_path):
            print(f"警告：注释文件 {csv_path} 不存在。无法加载脚本内容。")
            return

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)

            # 构建映射字典
            self.video_subjects = {
            row['id']: {
                'subject': row['subject'],
                'scene': row['scene']
            } 
            for index, row in df.iterrows()
        }

            print(f"成功加载 {len(self.video_subjects)} 个视频的脚本内容。")
        except Exception as e:
            print(f"错误：读取CSV文件时发生异常：{e}")

    def get_subject_scene(self, video_id):
        """
        根据视频ID获取对应的主题和场景。

        参数：
            video_id (str): 视频的ID。

        返回：
            str: 对应的视频主题和场景。如果视频ID不存在，返回空字符串。
        """
        subject_scene = self.video_subjects.get(video_id, None)
        subject = subject_scene['subject']
        scene = subject_scene['scene']
        return subject, scene
    
    def get_script(self, video_id):
        """
        根据视频ID获取对应的脚本内容。

        参数：
            video_id (str): 视频的ID。

        返回：
            str: 对应的视频脚本内容。如果视频ID不存在，返回空字符串。
        """
        script = self.video_scripts.get(video_id, None)
        return script

    def _fetch_object_classes(self):
        self.object_classes = [const.BACKGROUND]
        with open(os.path.join(self._data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE), 'r',
                  encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

    def _fetch_relationship_classes(self):
        self.relationship_classes = []
        with open(os.path.join(self._data_path, const.ANNOTATIONS, const.RELATIONSHIP_CLASSES_FILE), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'
        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]
        print('-------loading annotations---------slowly-----------')

    def _fetch_object_person_bboxes(self, datasize, filter_small_box=False):
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        if filter_small_box:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open('dataloader/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()

        if datasize == const.MINI:
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object

        return person_bbox, object_bbox

    def _fetch_valid_frames(self, person_bbox, object_bbox):
        video_dict = {}
        q = []
        for i in person_bbox.keys():
            if object_bbox[i][0][const.METADATA][const.SET] == self._phase:  # train or testing?
                video_name, frame_num = i.split('/')
                q.append(video_name)
                frame_valid = False
                for j in object_bbox[i]:  # the frame is valid if there is visible bbox
                    if j[const.VISIBLE]:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]
        return video_dict, q

    def fetch_video_data(self, index):
        frame_names = self._video_list[index]
        processed_ims = []
        im_scales = []
        for idx, name in enumerate(frame_names):
            im = cv2.imread(os.path.join(self._frames_path, name))  # channel h,w,3
            # im = im[:, :, ::-1]  # rgb -> bgr
            # cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000)
            im_scales.append(im_scale)
            processed_ims.append(im)
        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

    def _build_dataset(self, video_dict, person_bbox, object_bbox, all_video_names, filter_nonperson_box_frame=True):
        self._valid_video_names = []
        self._video_list = []
        self._video_size = []  # (w,h)
        self._gt_annotations = []
        self._non_gt_human_nums = 0
        self._non_heatmap_nums = 0
        self._non_person_video = 0
        self._one_frame_video = 0
        self._valid_nums = 0
        self._invalid_videos = []

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j][const.BOUNDING_BOX].shape[0] == 0:
                        self._non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self._valid_nums += 1

                gt_annotation_frame = [
                    {
                        const.PERSON_BOUNDING_BOX: person_bbox[j][const.BOUNDING_BOX],
                        const.FRAME: j
                    }
                ]

                # each frame's objects and human
                for k in object_bbox[j]:
                    if k[const.VISIBLE]:
                        assert k[const.BOUNDING_BOX] is not None, 'warning! The object is visible without bbox'
                        k[const.CLASS] = self.object_classes.index(k[const.CLASS])
                        # from xywh to xyxy
                        k[const.BOUNDING_BOX] = np.array([
                            k[const.BOUNDING_BOX][0], k[const.BOUNDING_BOX][1],
                            k[const.BOUNDING_BOX][0] + k[const.BOUNDING_BOX][2],
                            k[const.BOUNDING_BOX][1] + k[const.BOUNDING_BOX][3]
                        ])

                        k[const.ATTENTION_RELATIONSHIP] = torch.tensor(
                            [self.attention_relationships.index(r) for r in k[const.ATTENTION_RELATIONSHIP]],
                            dtype=torch.long)
                        k[const.SPATIAL_RELATIONSHIP] = torch.tensor(
                            [self.spatial_relationships.index(r) for r in k[const.SPATIAL_RELATIONSHIP]],
                            dtype=torch.long)
                        k[const.CONTACTING_RELATIONSHIP] = torch.tensor(
                            [self.contacting_relationships.index(r) for r in k[const.CONTACTING_RELATIONSHIP]],
                            dtype=torch.long)
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self._video_list.append(video)
                self._video_size.append(person_bbox[j][const.BOUNDING_BOX_SIZE])
                self._gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self._one_frame_video += 1
            else:
                self._non_person_video += 1

        print('x' * 60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self._video_list), self._valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self._non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self._one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self._non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self._video_list), self._valid_nums))
            print('{} frames have no human bbox in GT'.format(self._non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(
                self._non_heatmap_nums))
        print('x' * 60)

        self.invalid_video_names = np.setdiff1d(all_video_names, self._valid_video_names, assume_unique=False)

    def __len__(self):
        return len(self._video_list)

    @property
    def gt_annotations(self):
        return self._gt_annotations

    def __getitem__(self, index):
        frame_names = self._video_list[index]
        processed_ims = []
        im_scales = []
        for idx, name in enumerate(frame_names):
            im = cv2.imread(os.path.join(self._frames_path, name))  # channel h,w,3
            # im = im[:, :, ::-1]  # rgb -> bgr
            # cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000)
            im_scales.append(im_scale)
            processed_ims.append(im)
        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)
        if self.script_require:
            # 获取gt_annotation
            gt_annotation = self.gt_annotations[index]
            # 从gt_annotation中提取video_ids
            video_ids = {os.path.splitext(entry[0][const.FRAME].split('/')[0])[0] for entry in gt_annotation}
            # 假设每个index对应一个唯一的视频ID
            video_id = next(iter(video_ids)) if video_ids else ""
            script = self.get_script(video_id)
            return [img_tensor, im_info, gt_boxes, num_boxes, index], script, video_id  # 返回数据列表和脚本
        elif self.video_id_required:
            gt_annotation = self.gt_annotations[index]
            video_ids = {os.path.splitext(entry[0][const.FRAME].split('/')[0])[0] for entry in gt_annotation}
            # 假设每个index对应一个唯一的视频ID
            video_id = next(iter(video_ids)) if video_ids else ""
            return [img_tensor, im_info, gt_boxes, num_boxes, index], None, video_id  # 返回数据列表和脚本
        else:
            return [img_tensor, im_info, gt_boxes, num_boxes, index], None, None  # 返回数据列表和脚本


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
