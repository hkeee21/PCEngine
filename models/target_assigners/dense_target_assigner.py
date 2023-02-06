import copy
import torch
import numpy as np
from utils.iou3d.iou3d_utils import boxes_iou3d_gpu, boxes_iou_bev, box_transform_for_iou_calculation

__all__ = ['DenseTargetAssigner']


class DenseTargetAssigner(object):
    def __init__(self, 
                 box_coder,
                 negative_anchor_threshold: dict,
                 positive_anchor_threshold: dict,
                 use_multihead: bool = False):
        self.box_coder = box_coder
        self.negative_anchor_threshold = negative_anchor_threshold
        self.positive_anchor_threshold = positive_anchor_threshold
        self.use_multihead = use_multihead
        
    def assign_targets_single(self, 
                              anchors, 
                              gt_boxes, 
                              class_idx,
                              matched_threshold=0.6, 
                              unmatched_threshold=0.45):
        
        iou_anchors = box_transform_for_iou_calculation(anchors)
        iou_gt_boxes = box_transform_for_iou_calculation(gt_boxes)
        
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * (-class_idx)
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * (-1)
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = boxes_iou_bev(iou_anchors, iou_gt_boxes)
            anchor_to_gt_max, anchor_to_gt_argmax = torch.max(anchor_by_gt_overlap, 1)
            gt_to_anchor_max, gt_to_anchor_argmax = torch.max(anchor_by_gt_overlap, 0)
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -class_idx
            anchors_with_max_overlap = torch.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = class_idx #gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = class_idx #gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = torch.where(anchor_to_gt_max < unmatched_threshold)[0]        
        
        fg_inds = torch.where(labels > 0)[0]
        
        if len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            labels[anchors_with_max_overlap] = class_idx #gt_classes[gt_inds_force]
        
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)
        
        return {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets
        }
    
    def assign_targets(self, anchors, gt_boxes):
        """
        Args:
        anchors: dict of B x H x W x 2 x 8
        gt_boxes: dict of [(M, 7), ...]
        
        Returns:
        
        rpn_labels: dict of B x H x W x 16
        
        """
        
        num_classes = len(gt_boxes.keys())
        for key in gt_boxes:
            batch_size = len(gt_boxes[key])
            break
        H, W, _, num_anchors, _ = anchors[0].shape
        
        all_target_dict = []
        for k in range(batch_size):
            class_idx = 1
            target_list = []
            for c, cur_anchor in zip(list(gt_boxes.keys()), anchors):
                cur_anchors = cur_anchor.view(-1, self.box_coder.code_size)
                cur_gt_boxes = torch.from_numpy(gt_boxes[c][k]).to(cur_anchors.device)
                cur_gt_boxes = torch.cat([
                    cur_gt_boxes,
                    torch.zeros(
                        cur_gt_boxes.shape[0], 1, device=cur_anchors.device
                    ) + k
                ], -1)
                cur_anchors = torch.cat([
                    cur_anchors,
                    torch.zeros(
                        cur_anchors.shape[0], 1, device=cur_anchors.device
                    ) + k
                ], -1)
                cur_bbox_targets = self.assign_targets_single(
                    cur_anchors,
                    cur_gt_boxes,
                    class_idx,
                    self.positive_anchor_threshold[c],
                    self.negative_anchor_threshold[c]
                )
                target_list.append(cur_bbox_targets)
                class_idx += 1
            
            if not self.use_multihead:
                target_dict = {
                    'box_cls_labels': torch.cat([
                        t['box_cls_labels'].view(H, W, 1, num_anchors) for t in target_list
                    ], dim=-2).view(-1),
                    'box_reg_targets': torch.cat([
                        t['box_reg_targets'].view(H, W, 1, num_anchors, -1) for t in target_list
                    ], dim=-2).view(-1, self.box_coder.code_size),
                }
            else:
                target_dict = {
                    'box_cls_labels': torch.cat([
                        t['box_cls_labels'].view(H, W, 1, num_anchors) for t in target_list
                    ], dim=-2).permute(2, 3, 0, 1).contiguous().view(-1),
                    'box_reg_targets': torch.cat([
                        t['box_reg_targets'].view(H, W, 1, num_anchors, -1) for t in target_list
                    ], dim=-2).permute(2, 3, 0, 1, 4).contiguous().view(-1, self.box_coder.code_size),
                }
            all_target_dict.append(target_dict)
            
        
        rpn_labels = {
            'box_cls_labels': torch.cat([
                t['box_cls_labels'].unsqueeze(0) for t in all_target_dict
            ], dim=0),
            'box_reg_targets': torch.cat([
                t['box_reg_targets'].unsqueeze(0) for t in all_target_dict
            ], dim=0),
        }

        #print(torch.sum(rpn_labels['box_cls_labels'] > 0))

        return rpn_labels

    def __call__(self, anchors, gt_boxes):
        return self.assign_targets(anchors, gt_boxes)


if __name__ == '__main__':
    assigner = DenseTargetAssigner(10)
    from .dense_anchor_generator import DenseAnchorGenerator
    A = AnchorGenerator()
    #import pdb
    #pdb.set_trace()
    anchors = A.generate_anchors({
        'Car':
        torch.randn(2, 8, 200, 176).cuda(),
        'Pedestrian':
        torch.randn(2, 8, 200, 240).cuda(),
        'Cyclist':
        torch.randn(2, 8, 100, 120).cuda()
    })
    gt_boxes = {
        c: [np.random.randn(np.random.randint(10, 15), 7) for i in range(2)]
        for c in anchors.keys()
    }
    rpn_labels = assigner.assign_targets(anchors, gt_boxes)
    for c in rpn_labels:
        print(c, rpn_labels[c].shape)
