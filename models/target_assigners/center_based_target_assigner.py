import copy
import torch
import numpy as np
from utils.iou3d.iou3d_utils import boxes_iou3d_gpu, boxes_iou_bev
from typing import Union, List, Tuple
from script.utils import make_ntuple

__all__ = ['CenterBasedTargetAssigner']

# helper functions for heatmap generation #

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    
    top, bottom = min(y, radius), min(height - y, radius + 1)

    #masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    #masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    masked_heatmap = heatmap[x-left:x+right, y-top:y+bottom]
    masked_gaussian = gaussian[radius-left:radius+right, radius-top:radius+bottom]

    if masked_gaussian.shape == masked_heatmap.shape and min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

# end helper functions #

class CenterBasedTargetAssigner(object):
    def __init__(self, 
                 class_names: list,
                 grid_size: dict,
                 voxel_size: Union[np.ndarray, List[float], Tuple[float, ...], torch.Tensor],
                 proposal_stride: Union[int, List[int], Tuple[float, ...], torch.Tensor],
                 pc_area_scope: Union[np.ndarray, List[List[float]]], 
                 gaussian_overlap: float,
                 min_radius: float = 2,
                 use_multihead: bool = False,
                 **kwargs):
        self.gaussian_overlap = gaussian_overlap
        self.use_multihead = use_multihead
        self.grid_size = grid_size
        # voxel size: np.ndarray
        self.voxel_size = make_ntuple(voxel_size, 3)
        self.pc_area_scope = pc_area_scope
        self.proposal_stride = make_ntuple(proposal_stride, 3)
        self.min_radius = min_radius
        self.code_size = kwargs.get('code_size', 10)

        
    def assign_targets_single(self, 
                              gt_boxes, 
                              class_idx,
                              class_name,
                              box_cls_labels=None,
                              box_reg_targets=None,
                              box_reg_mask=None):

        num_objs = len(gt_boxes)
        sh, sw = self.proposal_stride[0], self.proposal_stride[2]
        
        grid_size = self.grid_size[class_name]
        H = int(round(grid_size[0] / sh))
        W = int(round(grid_size[1] / sw))
        box_cls_labels = np.zeros((H, W), dtype=np.float32)
        box_cls_mask = np.zeros((H, W), dtype=np.float32)
        if box_reg_targets is None:
          box_reg_targets = np.zeros((H, W, self.code_size), dtype=np.float32)
        if box_reg_mask is None:
          box_reg_mask = np.zeros((H, W), dtype=np.float32)

        for k in range(num_objs):
            h, w, l = gt_boxes[k][3], gt_boxes[k][4], gt_boxes[k][5]
            # KITTI coords system
            w, l = w / self.voxel_size[0] / sh, l / self.voxel_size[2] / sw
            if w > 0 and l > 0:
                radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                radius = max(int(self.min_radius), int(radius))

                # KITTI coords system
                x, z, y = gt_boxes[k][0], gt_boxes[k][1], gt_boxes[k][2]

                coor_x, coor_y = (x - self.pc_area_scope[0][0]) / self.voxel_size[0] / sh, \
                    (y - self.pc_area_scope[2][0]) / self.voxel_size[2] / sw

                ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                ct_int = ct.astype(np.int32)

                # out of feature map size
                if not (0 <= ct_int[0] < H and 0 <= ct_int[1] < W):
                    continue 

                draw_gaussian(box_cls_labels, ct, radius)

                # regression targets
                if self.code_size > 8:
                    # nuscenes
                    rot, vx, vy = gt_boxes[k][6:7], gt_boxes[k][7:8], gt_boxes[k][8:9]
                    box_reg_targets[ct_int[0], ct_int[1]] = np.concatenate(
                        [ct[0]-ct_int[0], z, ct[1]-ct_int[1], np.log(gt_boxes[k][3:6]), 
                        np.sin(rot), np.cos(rot), vx, vy], axis=None
                    )
                else:
                    rot = gt_boxes[k][6:7]
                    box_reg_targets[ct_int[0], ct_int[1]] = np.concatenate(
                        [ct[0]-ct_int[0], z, ct[1]-ct_int[1], np.log(gt_boxes[k][3:6]), 
                        np.sin(rot), np.cos(rot)], axis=None
                    )
                box_reg_mask[ct_int[0], ct_int[1]] = 1.
                box_cls_mask[ct_int[0], ct_int[1]] = 1

        """
        box_cls_labels = torch.from_numpy(box_cls_labels).cuda().view(-1, 1)
        box_reg_targets = torch.from_numpy(box_reg_targets).cuda().view(box_cls_labels.size(0), 1, -1)
        box_reg_mask = torch.from_numpy(box_reg_mask).cuda().view(-1, 1)
        """

        return dict(
            box_cls_labels=box_cls_labels,
            box_reg_targets=box_reg_targets,
            box_reg_mask=box_reg_mask,
            box_cls_mask=box_cls_mask
        )
    

    def assign_targets(self, gt_boxes):
        """
        Args:
        gt_boxes: dict of [(M, 7), ...]
        
        Returns:
        
        rpn_labels: dict of H x W x 16
        
        """
        
        num_classes = len(gt_boxes.keys())
        for key in gt_boxes:
            batch_size = len(gt_boxes[key])
            break
        
        all_target_dict = []
        for k in range(batch_size):
            class_idx = 1
            target_list = []
            cur_target_dict = dict()
            for c in gt_boxes:
                cur_target_dict = self.assign_targets_single(
                    gt_boxes[c][k],
                    class_idx,
                    c,
                    None,
                    cur_target_dict.get('box_reg_targets', None),
                    cur_target_dict.get('box_reg_mask', None)
                )
                target_list.append(cur_target_dict)
                
                class_idx += 1

            target_dict = dict(
                box_cls_labels=torch.cat([
                  torch.from_numpy(
                    t['box_cls_labels']
                  ).cuda().reshape(-1, 1) for t in target_list
                ], dim=-1),
                box_reg_targets=torch.from_numpy(
                  target_list[-1]['box_reg_targets']
                ).cuda().reshape(-1, 1, self.code_size),
                box_reg_mask=torch.from_numpy(
                  target_list[-1]['box_reg_mask']
                ).cuda().reshape(-1, 1),
                box_cls_mask=torch.cat([
                  torch.from_numpy(
                    t['box_cls_mask']
                  ).cuda().reshape(-1, 1) for t in target_list
                ], dim=-1),
            )
            all_target_dict.append(target_dict)
            
        
        rpn_labels = {
            'box_cls_labels': torch.cat([
                t['box_cls_labels'].unsqueeze(0) for t in all_target_dict
            ], dim=0),
            'box_reg_targets': torch.cat([
                t['box_reg_targets'].unsqueeze(0) for t in all_target_dict
            ], dim=0),
            'box_reg_mask': torch.cat([
                t['box_reg_mask'].unsqueeze(0) for t in all_target_dict
            ], dim=0),
            'box_cls_mask': torch.cat([
                t['box_cls_mask'].unsqueeze(0) for t in all_target_dict
            ], dim=0)
        }


        return rpn_labels

    def __call__(self, gt_boxes):
        return self.assign_targets(gt_boxes)


if __name__ == '__main__':
    pass
