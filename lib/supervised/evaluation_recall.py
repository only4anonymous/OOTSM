from functools import reduce

import numpy as np
import torch.nn as nn
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from collections import Counter
from lib.pytorch_misc import intersect_2d, argsort_desc


class BasicSceneGraphEvaluator:
    def __init__(
            self,
            mode,
            AG_object_classes,
            AG_all_predicates,
            AG_attention_predicates,
            AG_spatial_predicates,
            AG_contacting_predicates,
            iou_threshold=0.5,
            save_file="tmp",
            constraint=False,
            semi_threshold=None
    ):
        self.result_dict = {}
        self.mode = mode
        self.num_rel = len(AG_all_predicates)
        self.result_dict[self.mode + '_recall'] = {
            10: [], 20: [], 50: [], 100: []
        }
        self.result_dict[self.mode + '_mean_recall_collect'] = {
            k: [[] for _ in range(self.num_rel)] for k in (10, 20, 50, 100)
        }
        
        self.constraint = constraint  # semi constraint if True
        self.iou_threshold = iou_threshold
        self.AG_object_classes = AG_object_classes
        self.AG_all_predicates = AG_all_predicates
        self.AG_attention_predicates = AG_attention_predicates
        self.AG_spatial_predicates = AG_spatial_predicates
        self.AG_contacting_predicates = AG_contacting_predicates
        self.semi_threshold = semi_threshold
        self.save_file = save_file
    
    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        self.result_dict[self.mode + '_mean_recall_collect'] = {
            k: [[] for _ in range(self.num_rel)] for k in (10, 20, 50, 100)
        }
    
    def fetch_stats_json(self):
        recall_dict = {}
        mean_recall_dict = {}
        harmonic_mean_recall_dict = {}
        
        for k, v in self.result_dict[self.mode + '_recall'].items():
            recall_value = np.mean(v)
            recall_dict[k] = recall_value
        
        for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
            sum_recall = np.sum([np.mean(vi) if vi else 0.0 for vi in v])
            mean_recall_value = sum_recall / float(self.num_rel)
            mean_recall_dict[k] = mean_recall_value
        
        for k, recall_value in recall_dict.items():
            mean_recall_value = mean_recall_dict[k]
            harmonic_mean = 2 * mean_recall_value * recall_value / (mean_recall_value + recall_value)
            harmonic_mean_recall_dict[k] = harmonic_mean
        
        results = {
            "recall": recall_dict,
            "mean_recall": mean_recall_dict,
            "harmonic_mean_recall": harmonic_mean_recall_dict
        }
        
        return results
    
    def print_stats(self):
        def print_and_write(message):
            print(message)
            stats_file.write(message + '\n')
        
        with open(self.save_file, "a") as stats_file:
            header = f'======================{self.mode}======================'
            print_and_write(header)
            
            recall_dict = {}
            mean_recall_dict = {}
            harmonic_mean_recall_dict = {}
            
            for k, v in self.result_dict[self.mode + '_recall'].items():
                recall_value = np.mean(v)
                recall_dict[k] = recall_value
                print_and_write(f'R@{k}: {recall_value:.6f}')
            
            for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
                sum_recall = np.sum([np.mean(vi) if vi else 0.0 for vi in v])
                mean_recall_value = sum_recall / float(self.num_rel)
                mean_recall_dict[k] = mean_recall_value
                print_and_write(f'mR@{k}: {mean_recall_value:.6f}')
            
            for k, recall_value in recall_dict.items():
                mean_recall_value = mean_recall_dict[k]
                harmonic_mean = 2 * mean_recall_value * recall_value / (mean_recall_value + recall_value)
                harmonic_mean_recall_dict[k] = harmonic_mean
                print_and_write(f'hR@{k}: {harmonic_mean:.6f}')
    
    def fetch_pred_tuples(self, gt, pred):
        idx_pred_triplets_map = {}
        pred['attention_distribution'] = nn.functional.softmax(pred['attention_distribution'], dim=1)
        for idx, frame_gt in enumerate(gt):
            frame_idx = frame_gt[0]['frame'].split('/')[-1].split('.')[0]
            # first part for attention and contact, second for spatial
            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),  # attention
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:, ::-1],  # spatial
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()),
                                    axis=0)  # contacting
            
            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                 np.zeros(
                     [pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])),
                axis=1)
            pred_scores_3 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                 pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)
            
            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            
            pred_rel_inds = pred_entry['pred_rel_inds']
            rel_scores = pred_entry['rel_scores']
            
            pred_boxes = pred_entry['pred_boxes'].astype(float)
            pred_classes = pred_entry['pred_classes']
            obj_scores = pred_entry['obj_scores']
            
            if self.constraint == 'no':
                obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
                overall_scores = obj_scores_per_rel[:, None] * rel_scores
                score_inds = argsort_desc(overall_scores)[:100]
                pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
                predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
            else:
                pred_rels = np.column_stack(
                    (pred_rel_inds, rel_scores.argmax(1)))  # 1+  dont add 1 because no dummy 'no relations'
                predicate_scores = rel_scores.max(1)
            
            if pred_rels.size == 0:
                continue
            
            pred_triplets, pred_triplet_boxes, relation_scores = \
                _triplet(pred_rels[:, 2], pred_rels[:, :2], pred_classes, pred_boxes,
                         predicate_scores, obj_scores)
            
            sorted_scores = relation_scores.prod(1)
            pred_triplets = pred_triplets[sorted_scores.argsort()[::-1], :]
            
            # Subject Object Relationship Class
            idx_pred_triplets_map[frame_idx] = pred_triplets[:, [0, 2, 1]]
        
        return idx_pred_triplets_map
    
    def evaluate_scene_graph(self, gt, pred):
        """collect the ground truth and prediction"""
        pred['attention_distribution'] = nn.functional.softmax(pred['attention_distribution'], dim=1)
        for idx, frame_gt in enumerate(gt):
            # generate the ground truth
            gt_boxes = np.zeros([len(frame_gt), 4])  # now there is no person box! we assume that person box index == 0
            gt_classes = np.zeros(len(frame_gt))
            gt_relations = []
            # 修改后的版本：存列表，保留所有重复标签
            gt_relations_attention = {}
            gt_relations_spatial = {}
            gt_relations_contact = {}
            human_idx = 0
            gt_classes[human_idx] = 1
            gt_boxes[human_idx] = frame_gt[0]['person_bbox']
            for m, n in enumerate(frame_gt[1:]):
                gt_boxes[m + 1, :] = n['bbox']
                gt_classes[m + 1] = n['class']
                # Attention triplet：格式为 [human_idx, m+1, ...]
                att_idx = self.AG_all_predicates.index(self.AG_attention_predicates[n['attention_relationship']])
                gt_relations.append([human_idx, m + 1, att_idx])
                key_attn = (human_idx, m + 1)
                # 存储 attention 标签列表（可能重复）
                att_list = (n['attention_relationship'].numpy().tolist() 
                            if hasattr(n['attention_relationship'], 'numpy') 
                            else [n['attention_relationship']])
                # 用 setdefault 保证 key 存在，然后 extend 列表
                gt_relations_attention.setdefault(key_attn, []).extend(att_list)
                
                # Spatial triplet：格式为 [m+1, human_idx, ...]
                for spatial in n['spatial_relationship'].numpy().tolist():
                    spat_idx = self.AG_all_predicates.index(self.AG_spatial_predicates[spatial])
                    gt_relations.append([m + 1, human_idx, spat_idx])
                    key_spat = (m + 1, human_idx)  # 预测时通常将 spatial 对反转为 (human_idx, m+1)
                    gt_relations_spatial.setdefault(key_spat, []).append(spatial)
                
                # Contact triplet：格式为 [human_idx, m+1, ...]
                for contact in n['contacting_relationship'].numpy().tolist():
                    cont_idx = self.AG_all_predicates.index(self.AG_contacting_predicates[contact])
                    gt_relations.append([human_idx, m + 1, cont_idx])
                    key_cont = (human_idx, m + 1)
                    gt_relations_contact.setdefault(key_cont, []).append(contact)
            
            gt_entry = {
                'gt_classes': gt_classes,
                'gt_relations': np.array(gt_relations),
                'gt_boxes': gt_boxes,
                'gt_relations_attention': gt_relations_attention,
                'gt_relations_spatial': gt_relations_spatial,
                'gt_relations_contact': gt_relations_contact,
            }
            
            # first part for attention and contact, second for spatial
            try:
                rels_i = np.concatenate((
                    pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),
                    pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:, ::-1],
                    pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()),
                    axis=0)
            except Exception as e:
                print(f"Error processing frame {idx}: {str(e)}")
                print(f"\n=== Debug Info for frame {idx} ===")
                print(f"pair_idx shape: {pred['pair_idx'].shape}")
                print(f"pair_idx data:\n{pred['pair_idx']}")
                print(f"im_idx shape: {pred['im_idx'].shape}")
                print(f"im_idx data:\n{pred['im_idx']}")
                print(f"Mask result shape: {(pred['im_idx'] == idx).shape}")
                print(f"Filtered pair_idx shape: {pred['pair_idx'][pred['im_idx'] == idx].shape}")
                continue
            
            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                 np.zeros(
                     [pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])),
                axis=1)
            pred_scores_3 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                 pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)
            
            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            
            evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                               iou_thresh=self.iou_threshold, method=self.constraint, threshold=self.semi_threshold,
                               num_rel=self.num_rel)


def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, method=None, threshold=0.9, num_rel=26, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param mode:
    :param num_rel:
    :param threshold:
    :param method:
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']
    
    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']
    
    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']
    
    if method == 'semi':
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            if rel_scores[i, 0] + rel_scores[i, 1] > 0:
                # this is the attention distribution
                pred_rels.append(np.append(j, rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            elif rel_scores[i, 3] + rel_scores[i, 4] > 0:
                # this is the spatial distribution
                for k in np.where(rel_scores[i] > threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i, k])
            elif rel_scores[i, 9] + rel_scores[i, 10] > 0:
                # this is the contact distribution
                for k in np.where(rel_scores[i] > threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i, k])
        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
    else:
        # pred_rels = np.column_stack(
        #     (pred_rel_inds, rel_scores.argmax(1)))  # 1+  dont add 1 because no dummy 'no relations'
        # predicate_scores = rel_scores.max(1)
        # 默认方法：按照三个分支分别处理
        N = pred_rel_inds.shape[0] // 3
        attn_pred_rel_inds = pred_rel_inds[:N]
        spat_pred_rel_inds = pred_rel_inds[N:2*N]
        cont_pred_rel_inds = pred_rel_inds[2*N:3*N]
        
        attn_rel_scores = rel_scores[:N]
        spat_rel_scores = rel_scores[N:2*N]
        cont_rel_scores = rel_scores[2*N:3*N]
        
        # 从 gt_entry 中获取三个字典
        gt_relations_attention = gt_entry.get('gt_relations_attention', {})
        gt_relations_spatial = gt_entry.get('gt_relations_spatial', {})
        gt_relations_contact = gt_entry.get('gt_relations_contact', {})
        
        all_pred_rels = []
        all_pred_scores = []
        
        # Attention 分支处理
        for i in range(N):
            pair = tuple(int(x) - min(attn_pred_rel_inds[i]) for x in attn_pred_rel_inds[i])  # 例如 (0, X)
            K = len(gt_relations_attention.get(pair, []))
            topK = np.argsort(-attn_rel_scores[i])[:K]
            for idx in topK:
                new_triplet = np.concatenate((attn_pred_rel_inds[i], [idx]))
                all_pred_rels.append(new_triplet)
                all_pred_scores.append(attn_rel_scores[i, idx])
        
        # Spatial 分支处理
        for i in range(N):
            pair = tuple(int(x) - min(spat_pred_rel_inds[i]) for x in spat_pred_rel_inds[i])  # 预测时 spatial 分支预测时通常反转，所以 ground truth key与 attention 相同
            K = len(gt_relations_spatial.get(pair, []))
            topK = np.argsort(-spat_rel_scores[i])[:K]
            for idx in topK:
                new_triplet = np.concatenate((spat_pred_rel_inds[i], [idx]))
                all_pred_rels.append(new_triplet)
                all_pred_scores.append(spat_rel_scores[i, idx])
        
        # Contacting 分支处理
        for i in range(N):
            pair = tuple(int(x) - min(spat_pred_rel_inds[i]) for x in cont_pred_rel_inds[i])
            K = len(gt_relations_contact.get(pair, []))
            topK = np.argsort(-cont_rel_scores[i])[:K]
            for idx in topK:
                new_triplet = np.concatenate((cont_pred_rel_inds[i], [idx]))
                all_pred_rels.append(new_triplet)
                all_pred_scores.append(cont_rel_scores[i, idx])
        
        pred_rels = np.array(all_pred_rels)
        predicate_scores = np.array(all_pred_scores)
        
    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
        gt_rels, gt_boxes, gt_classes,
        pred_rels, pred_boxes, pred_classes,
        predicate_scores, obj_scores, phrdet=mode == 'phrdet', **kwargs)
    
    for k in result_dict[mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt[:k])
        recall_hit = [0] * num_rel
        recall_count = [0] * num_rel
        
        for idx in range(gt_rels.shape[0]):
            local_label = gt_rels[idx, 2]
            recall_count[int(local_label)] += 1
        
        for idx in range(len(match)):
            local_label = gt_rels[int(match[idx]), 2]
            recall_hit[int(local_label)] += 1
        
        for n in range(num_rel):
            if recall_count[n] > 0:
                result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
        
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, pred_5ples, rel_scores


###########################
def evaluate_recall(
        gt_rels,
        gt_boxes,
        gt_classes,
        pred_rels,
        pred_boxes,
        pred_classes,
        rel_scores=None,
        cls_scores=None,
        iou_thresh=0.5,
        phrdet=False
):
    """
    Evaluates the recall
    :param cls_scores:
    :param rel_scores:
    :param iou_thresh:
    :param phrdet:
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)
    
    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0
    
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    
    try:
        assert pred_rels[:, :2].max() < pred_classes.shape[0]
    except AssertionError:
        print("assert error ")
    # pdb.set_trace()
    
    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
    # assert np.all(pred_rels[:,2] > 0)
    
    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:, 2], pred_rels[:, :2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)
    
    # sorted_scores = relation_scores.prod(1)
    # pred_triplets = pred_triplets[sorted_scores.argsort()[::-1], :]
    # pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1], :]
    # relation_scores = relation_scores[sorted_scores.argsort()[::-1], :]
    # scores_overall = relation_scores.prod(1)
    
    # if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
    #     print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
    # pdb.set_trace()
    # raise ValueError("Somehow the relations weren't sorted properly")
    
    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )
    
    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:, :2],
        pred_triplets[:, [0, 2, 1]],
    ))
    
    return pred_to_gt, pred_5ples, relation_scores


def _triplet(
        predicates,
        relations,
        classes,
        boxes,
        predicate_scores=None,
        class_scores=None
):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])
    
    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))
    
    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))
    
    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_boxes,
        pred_boxes,
        iou_thresh,
        phrdet=False
):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)
            
            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)
            
            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh
        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
            
            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
        
        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
