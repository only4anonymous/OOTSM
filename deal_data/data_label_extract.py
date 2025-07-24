#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import sys
from tqdm import tqdm

# 添加项目根目录到路径
project_root = "your/project/path"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataloader.action_genome.ag_dataset import AG

# 关系类别常量
ATTN_REL_CLASSES = ['looking_at', 'not_looking_at', 'unsure']
SPAT_REL_CLASSES = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']
CONT_REL_CLASSES = [
    'carrying', 'covered_by', 'drinking_from', 'eating', 'have_it_on_the_back',
    'holding', 'leaning_on', 'lying_on', 'not_contacting', 'other_relationship',
    'sitting_on', 'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on'
]

def extract_frame_number(frame_info):
    """从帧信息中提取帧号"""
    try:
        return int(frame_info.split('/')[-1].split('.')[0])
    except:
        return 0

def format_object_list(objects):
    """格式化对象列表为自然语言字符串"""
    if len(objects) == 1:
        return f"the {objects[0]}"
    elif len(objects) == 2:
        return f"the {objects[0]} and the {objects[1]}"
    else:
        return ", ".join([f"the {obj}" for obj in objects[:-1]]) + f", and the {objects[-1]}"

def generate_script_for_frame(objects):
    """
    为一帧生成自然语言脚本描述
    
    Args:
        objects: 包含对象及其关系的字典
        
    Returns:
        str: 生成的自然语言脚本
    """
    descriptions = []
    
    # 1. 生成注意力描述
    looking_at_objects = [obj for obj, rels in objects.items() 
                          if 'attention' in rels and 'looking_at' in rels['attention']]
    not_looking_at_objects = [obj for obj, rels in objects.items() 
                             if 'attention' in rels and 'not_looking_at' in rels['attention']]
    
    if looking_at_objects:
        looking_at_text = format_object_list(looking_at_objects)
        descriptions.append(f"The person is looking at {looking_at_text}.")
    if not_looking_at_objects:
        not_looking_at_text = format_object_list(not_looking_at_objects)
        descriptions.append(f"The person is not looking at {not_looking_at_text}.")
    
    # 2. 生成空间关系描述
    spatial_descriptions = {}
    for obj, rels in objects.items():
        if 'spatial' in rels and rels['spatial']:
            for spatial_rel in rels['spatial']:
                if spatial_rel != "None":
                    spatial_key = spatial_rel.replace('_', ' ')
                    if spatial_key not in spatial_descriptions:
                        spatial_descriptions[spatial_key] = []
                    spatial_descriptions[spatial_key].append(obj)
    
    for spat_rel, obj_list in spatial_descriptions.items():
        obj_text = format_object_list(obj_list)
        be_verb = "are" if " and " in obj_text else "is"
        descriptions.append(f"{obj_text.capitalize()} {be_verb} {spat_rel} the person.")
    
    # 3. 生成接触关系描述
    contact_descriptions = {}
    for obj, rels in objects.items():
        if 'contact' in rels and rels['contact']:
            valid_contacts = [rel for rel in rels['contact'] if rel != 'other_relationship' and rel != "None"]
            for contact_rel in valid_contacts:
                contact_key = contact_rel.replace('_', ' ')
                if contact_key not in contact_descriptions:
                    contact_descriptions[contact_key] = []
                contact_descriptions[contact_key].append(obj)
    
    for cont_rel, obj_list in contact_descriptions.items():
        obj_text = format_object_list(obj_list)
        descriptions.append(f"The person is {cont_rel} {obj_text}.")
    
    if not descriptions:
        return "No notable interactions in this frame."
    
    return " ".join(descriptions)

def get_video_id_from_path(frame_path):
    """从帧路径中提取视频ID"""
    video_name = frame_path.split('/')[0]
    return os.path.splitext(video_name)[0]

def extract_scene_graph(ag_dataset, target_video_id):
    """
    提取指定视频ID的所有帧的场景图信息，并为每帧生成自然语言脚本
    
    Args:
        ag_dataset: Action Genome数据集实例
        target_video_id: 要提取场景图信息的视频ID
        
    Returns:
        dict: 包含场景图信息的字典
    """
    # 初始化结果字典
    result = {}
    
    # 查找视频在数据集中的索引
    video_idx = -1
    gt_annotations = ag_dataset.gt_annotations
    
    for idx, anno in enumerate(gt_annotations):
        if not anno:  # 跳过空注释
            continue
        # 从第一帧的路径中提取视频ID
        frame_path = anno[0][0]['frame']
        video_id = get_video_id_from_path(frame_path)
        
        if video_id == target_video_id:
            video_idx = idx
            break
    
    if video_idx == -1:
        print(f"错误: 未找到ID为{target_video_id}的视频!")
        return {}
    
    # 获取视频的注释信息
    gt_anno_video = gt_annotations[video_idx]
    
    # 处理每一帧
    frame_dict = {}
    for frame_data in gt_anno_video:
        if not frame_data:
            continue
            
        # 获取帧号
        frame_info = frame_data[0]
        frame_id = extract_frame_number(frame_info['frame'])
        
        # 初始化当前帧的场景图信息
        frame_scene_graph = {
            "objects": [],
            "attention": {},
            "spatial": {},
            "contact": {}
        }
        
        # 处理每个物体
        for obj_idx, obj_dict in enumerate(frame_data[1:], 1):
            # 获取物体类别
            cls_idx = obj_dict.get('class', -1)
            if 0 <= cls_idx < len(ag_dataset.object_classes):
                obj_name = ag_dataset.object_classes[cls_idx]
            else:
                obj_name = "unknown"
                
            frame_scene_graph["objects"].append(obj_name)
            
            # 获取注意力关系
            attn_ids = obj_dict.get('attention_relationship', [])
            if hasattr(attn_ids, 'tolist'):
                attn_ids = attn_ids.tolist()
            attn_relations = [ATTN_REL_CLASSES[i] for i in attn_ids] if attn_ids else []
            
            # 获取空间关系
            spat_ids = obj_dict.get('spatial_relationship', [])
            if hasattr(spat_ids, 'tolist'):
                spat_ids = spat_ids.tolist()
            spat_relations = [SPAT_REL_CLASSES[i] for i in spat_ids] if spat_ids else []
            
            # 获取接触关系
            cont_ids = obj_dict.get('contacting_relationship', [])
            if hasattr(cont_ids, 'tolist'):
                cont_ids = cont_ids.tolist()
            cont_relations = [CONT_REL_CLASSES[i] for i in cont_ids] if cont_ids else []
            
            # 将关系添加到场景图中
            frame_scene_graph["attention"][obj_name] = attn_relations
            frame_scene_graph["spatial"][obj_name] = spat_relations
            frame_scene_graph["contact"][obj_name] = cont_relations
        
        # 生成自然语言脚本
        script_objects = {}
        for obj_name in frame_scene_graph["objects"]:
            script_objects[obj_name] = {
                "attention": frame_scene_graph["attention"].get(obj_name, []),
                "spatial": frame_scene_graph["spatial"].get(obj_name, []),
                "contact": frame_scene_graph["contact"].get(obj_name, [])
            }
        
        script = generate_script_for_frame(script_objects)
        frame_scene_graph["script"] = script
        
        # 将当前帧的场景图添加到结果字典中
        frame_dict[str(frame_id)] = frame_scene_graph
    
    return {str(target_video_id): frame_dict}

def load_existing_scene_graph(output_file):
    """
    加载已存在的场景图JSON文件
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        dict: 加载的场景图信息，如果文件不存在则返回空字典
    """
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取现有JSON文件时出错: {e}")
            return {}
    return {}

def save_scene_graph_to_json(scene_graph_dict, output_file):
    """
    将场景图信息保存为JSON文件
    
    Args:
        scene_graph_dict: 包含场景图信息的字典
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scene_graph_dict, f, indent=2, ensure_ascii=False)
    print(f"场景图信息已保存至 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='提取并保存指定视频ID列表的所有帧的场景图信息和自然语言脚本')
    parser.add_argument('--video_ids', type=str, nargs='+', required=True, help='要提取场景图的视频ID列表，多个ID用空格分隔')
    parser.add_argument('--data_path', type=str, required=True, help='Action Genome数据集路径')
    parser.add_argument('--output', type=str, default='scene_graph.json', help='输出JSON文件路径')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'val'], help='数据集分区')
    parser.add_argument('--datasize', type=str, default='full', choices=['mini', 'full'], help='数据集大小')
    args = parser.parse_args()
    
    # 加载Action Genome数据集
    print(f"加载Action Genome数据集 (phase={args.phase}, datasize={args.datasize})...")
    ag_dataset = AG(
        phase=args.phase,
        datasize=args.datasize,
        data_path=args.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        script_require=False,
        video_id_required=True
    )
    
    # 检查输出文件是否已存在，如存在则加载
    existing_data = load_existing_scene_graph(args.output)
    
    # 提取场景图信息
    all_results = existing_data
    for video_id in tqdm(args.video_ids, desc="处理视频"):
        print(f"正在提取视频ID {video_id} 的场景图信息...")
        
        # 检查是否已经处理过该视频
        if str(video_id) in all_results:
            print(f"视频ID {video_id} 已存在于输出文件中，跳过...")
            continue
            
        scene_graph_dict = extract_scene_graph(ag_dataset, video_id)
        
        # 合并结果
        if scene_graph_dict:
            all_results.update(scene_graph_dict)
    
    # 保存为JSON文件
    print(f"正在保存 {len(all_results)} 个视频的场景图信息...")
    save_scene_graph_to_json(all_results, args.output)

if __name__ == "__main__":
    main()