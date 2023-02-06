import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Union

from .anchor_head_template import AnchorHeadTemplate

__all__ = ['AnchorHead']

class AnchorHead(AnchorHeadTemplate):
    def __init__(self, 
                 input_channels: int,
                 num_class: int, 
                 class_names: List[str], 
                 grid_size: Union[List[int], Tuple[int, ...]], 
                 pc_area_scope: Union[List[List[int]], np.ndarray], 
                 negative_threshold: dict,
                 positive_threshold: dict,
                 mean_size: dict,
                 mean_center_z: dict,
                 **kwargs):
        super().__init__(
            num_class=num_class, class_names=class_names,
            grid_size=grid_size, pc_area_scope=pc_area_scope,
            negative_threshold=negative_threshold, positive_threshold=positive_threshold,
            mean_size=mean_size, mean_center_z=mean_center_z, **kwargs
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if kwargs.get('use_direction_classifier', False):
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.num_dir_bins,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, inputs, gt_boxes=None):
        outputs = dict()
        targets = dict()
        
        cls_preds = self.conv_cls(inputs)
        box_preds = self.conv_box(inputs)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        outputs.update(cls_preds=cls_preds)
        outputs.update(box_preds=box_preds)
        #print(torch.sum(torch.sigmoid(cls_preds.max(-1).values) > 0.1))

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(inputs)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            outputs.update(dir_cls_preds=dir_cls_preds)
        else:
            dir_cls_preds = None
        
        
        if self.training and gt_boxes is not None:
            targets_dict = self.assign_targets(
                gt_boxes=gt_boxes
            )
            targets.update(targets_dict)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            outputs['batch_cls_preds'] = batch_cls_preds
            outputs['batch_box_preds'] = batch_box_preds
            #outputs['cls_preds_normalized'] = False
        return outputs, targets
