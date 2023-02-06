import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..criterions import SigmoidFocalLoss, WeightedSmoothL1Loss
from utils import common_utils
from utils.kitti.box_utils import BoxCoder
from ..target_assigners import DenseAnchorGenerator, DenseTargetAssigner
from typing import Union, List, Tuple

__all__ = ['AnchorHeadTemplate']


class AnchorHeadTemplate(nn.Module):
    def __init__(self, 
                 num_class: int, 
                 class_names: List[str], 
                 grid_size: Union[List[float], Tuple[float, ...], np.ndarray], 
                 pc_area_scope: Union[List[List[float]], np.ndarray], 
                 negative_threshold: dict,
                 positive_threshold: dict,
                 mean_size: dict,
                 mean_center_z: dict,
                 proposal_stage: int,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.class_names = class_names
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold
        self.mean_size = mean_size
        self.mean_center_z = mean_center_z
        self.proposal_stage = proposal_stage
        self.use_multihead = kwargs.get('use_multihead', False)
        self.cls_weight = kwargs.get('cls_weight', 1.0)
        self.reg_weight = kwargs.get('reg_weight', 2.0)
        self.dir_weight = kwargs.get('dir_weight', 0.2)
        self.pos_cls_weight = kwargs.get('pos_cls_weight', 1.0)
        self.neg_cls_weight = kwargs.get('neg_cls_weight', 1.0)
        self.dir_offset = kwargs.get('dir_offset', np.pi / 4.)
        self.dir_limit_offset = kwargs.get('dir_limit_offset', 0.0)
        self.num_dir_bins = kwargs.get('num_dir_bins', 2)
        code_cfg = kwargs.get('code_cfg', None)

        self.box_coder = BoxCoder(**code_cfg)
        self.loss_cfg = kwargs.get('loss_cfg', {'beta': 1./9.,
            'code_weight': [1. for i in range(self.box_coder.code_size)]})

        self.anchors, self.num_anchors_per_location = self.generate_anchors(
            grid_size=grid_size, pc_area_scope=pc_area_scope,
            mean_size=mean_size, mean_center_z=mean_center_z,
            proposal_stage=self.proposal_stage, anchor_ndim=self.box_coder.code_size
        )
        
        self.target_assigner = self.get_target_assigner(negative_threshold, positive_threshold)

        self.build_losses()

    @staticmethod
    def generate_anchors(grid_size, pc_area_scope,
                         mean_size, mean_center_z, proposal_stage,
                         anchor_ndim=7):
        
        anchor_generator = DenseAnchorGenerator(
            grid_size=np.array(
                grid_size, 'int'
            )[:2] // 2 ** proposal_stage,
            pc_area_scope=pc_area_scope,
            mean_size=mean_size,
            mean_center_z=mean_center_z#,
            #rotations=rotations
        )
        
        anchors_list, num_anchors_per_location = anchor_generator.generate_anchors()
        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location

    def get_target_assigner(self, negative_anchor_threshold, positive_anchor_threshold):
        target_assigner = DenseTargetAssigner(
            self.box_coder,
            negative_anchor_threshold,
            positive_anchor_threshold,
            use_multihead=self.use_multihead
        )
        return target_assigner

    def build_losses(self):
        self.add_module(
            'cls_loss_func',
            SigmoidFocalLoss()
        )
        self.add_module(
            'reg_loss_func',
            WeightedSmoothL1Loss(
                **self.loss_cfg
            )
        )
        self.dir_loss_func = functools.partial(F.cross_entropy, reduction='none')
        

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self, inputs, targets):
        cls_preds = inputs['cls_preds']
        box_cls_labels = targets['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0

        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        
        #print(torch.sum(torch.sigmoid(cls_preds.max(-1).values) > 0.1))
        #print(torch.sum(torch.sigmoid(cls_preds[box_cls_labels > 0].max(-1).values) > 0.1))
        
        one_hot_targets = one_hot_targets[..., 1:]
        
        # [N, M]
        cls_loss_src = self.cls_loss_func(
            cls_preds, one_hot_targets, weights=cls_weights.unsqueeze(-1)
        ) 
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.cls_weight
        return cls_loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=False, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        #print(reg_targets[..., 6].max(), reg_targets[..., 6].min(), rot_gt.max(), rot_gt.min())
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self, inputs, targets):
        box_preds = inputs['box_preds']
        box_dir_cls_preds = inputs.get('dir_cls_preds', None)
        box_reg_targets = targets['box_reg_targets']
        box_cls_labels = targets['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            """
            if self.use_multihead:
                assert 0
            else:
                # [H x W x 1 x #rot x 7] x 3
                anchors = torch.cat(self.anchors, dim=-3)
            """
            anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(
            batch_size, -1,
            box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
            box_preds.shape[-1]
        )
                
        #print('PRED', box_preds[positives])
        #print('TGT', box_reg_targets[positives])
        
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(
            box_preds, box_reg_targets
        )
        # [N, M]
        reg_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin
        )  
        reg_loss_src *= reg_weights.unsqueeze(-1)
        reg_loss = reg_loss_src.sum() / batch_size

        reg_loss = reg_loss * self.reg_weight

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.dir_offset,
                num_bins=self.num_dir_bins
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.num_dir_bins)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(
                dir_logits.transpose(2, 1), dir_targets)
            dir_loss *= weights
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.dir_weight
            #print(dir_loss)
            #print(torch.bincount(dir_targets[positives]))
            #print(torch.sum(dir_logits.argmax(-1)==dir_targets), dir_targets.numel())
        else:
            dir_loss = torch.zeros_like(reg_loss)
        return reg_loss, dir_loss

    def get_loss(self, inputs, targets):
        cls_loss = self.get_cls_layer_loss(inputs, targets)
        reg_loss, dir_loss = self.get_box_reg_layer_loss(inputs, targets)

        return {
            'rpn_loss_cls': cls_loss,
            'rpn_loss_loc': reg_loss,
            'rpn_loss_dir': dir_loss
        }

    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(self.anchors, dim=-3)
                anchors = anchors.permute(2, 3, 0, 1, 4).contiguous()
            else:
                # H x W x 1 x #rot x 7
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        
        if isinstance(cls_preds, list):
            batch_size = cls_preds[0].shape[0]
        else:
            batch_size = cls_preds.shape[0]
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) \
            if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.dir_offset
            dir_limit_offset = self.dir_limit_offset
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.num_dir_bins)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, 0.0, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)
            
            #print(batch_box_preds[..., 6].max(), batch_box_preds[..., 6].min())
        
        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
