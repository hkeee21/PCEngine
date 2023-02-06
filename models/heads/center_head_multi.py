import numpy as np
import torch
import torch.nn as nn
from ..criterions import FocalLossCenterPoint, L1LossCenterPoint
from ..modules import ConvBlock
from .anchor_head_multi import SingleHead
from ..target_assigners import CenterBasedTargetAssigner
from utils.iou3d import iou3d_utils
from typing import Union, List, Tuple
# from torchsparse.utils import make_ntuple
from script.utils import make_ntuple


__all__ = ['CenterHeadMulti']

class CenterSingleHead(SingleHead):
    def __init__(self, 
                 input_channels: int,
                 last_kernel_size: int,
                 num_class: int, 
                 code_size: int,
                 reg_config: dict,
                 head_label_indices: torch.Tensor,
                 separate_reg: bool = True,
                 use_multihead: bool = True,
                 **kwargs):
        super().__init__(
            input_channels=input_channels,
            last_kernel_size=last_kernel_size,
            num_class=num_class,
            code_size=code_size,
            num_anchors_per_location=1,
            reg_config=reg_config,
            head_label_indices=head_label_indices,
            separate_reg=True,
            use_multihead=True,
            **kwargs
        )
    
    def init_weights(self):
        for m in self.conv_box.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        offset = -2.19
        if isinstance(self.conv_cls, nn.Conv2d):
            nn.init.constant_(self.conv_cls.bias, offset)
        else:
            nn.init.constant_(self.conv_cls[-1].bias, offset)


# TBD: reformat the code (currently highly overlaps with AnchorHeadMulti)
class CenterHeadMulti(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 last_kernel_size: int,
                 num_class: int, 
                 class_names: List[str], 
                 voxel_size: Union[np.ndarray, List[float], Tuple[float, ...], torch.Tensor],
                 proposal_stride: Union[int, List[int], Tuple[float, ...], torch.Tensor],
                 grid_size: dict, 
                 pc_area_scope: Union[np.ndarray, List[List[float]]], 
                 reg_config: dict,
                 rpn_head_configs: dict,
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.class_names = class_names
        self.voxel_size = tuple(voxel_size)
        self.proposal_stride = make_ntuple(proposal_stride, 3)
        self.grid_size = grid_size
        self.pc_area_scope = pc_area_scope
        self.reg_config = reg_config
        self.rpn_head_configs = rpn_head_configs
        self.last_kernel_size = last_kernel_size
        self.cls_weight = kwargs.get('cls_weight', 1.0)
        self.reg_weight = kwargs.get('reg_weight', 2.0)
        self.dir_weight = kwargs.get('dir_weight', 0.2)
        self.assigner_cfg = kwargs.get('assigner_cfg', dict(
            gaussian_overlap=0.1,
            min_radius=2.0
        ))
        self.box_coder_cfg = kwargs.get('code_cfg', dict(
            code_size=9,
            encode_angle_by_sincos=True
        ))
        self.code_size = self.box_coder_cfg['code_size'] + int(self.box_coder_cfg['encode_angle_by_sincos'])
        self.loss_cfg = kwargs.get('loss_cfg', {'beta': 1./9.,
            'code_weight': [1. for i in range(10)]})
        shared_channels = self.reg_config.get('shared_channels', None)
        # shared convolution layers
        if shared_channels is not None:
            self.shared_conv = ConvBlock(
                input_channels, shared_channels, 
                kernel_size=3, stride=1, padding=1, bias=False,
                eps=0.001, momentum=0.01
            )
        else:
            self.shared_conv = None
            shared_channels = input_channels
        
        self.use_direction_classifier = kwargs.get(
            'use_direction_classifier', False
        )
        self.use_iou_head = kwargs.get(
            'use_iou_head', False
        )
        self.rpn_heads = None
        self.make_multiheads(shared_channels)
        self.build_losses()
        self.use_multihead = True
        self.target_assigner = self.get_target_assigner()


    def build_losses(self):
        self.add_module(
            'cls_loss_func',
            FocalLossCenterPoint()
        )
        self.add_module(
            'reg_loss_func',
            L1LossCenterPoint(
                **self.loss_cfg
            )
        )


    def make_multiheads(self, input_channels):
        rpn_heads = []
        class_names = []
        reg_config = self.reg_config
        # example for reg_config:
        # {'reg_list': 
        #   { 'reg': 2, 'height': 1, 'size': 3, 'angle': 2, 'velo': 2},
        # 'mid_channels': 64, 'num_middle_layers': 1
        # }
        for rpn_head_cfg in self.rpn_head_configs:
            class_names += rpn_head_cfg['head_cls_name']
        for rpn_head_cfg in self.rpn_head_configs:
            head_label_indices = torch.from_numpy(np.array([
                class_names.index(cur_name) + 1 \
                for cur_name in rpn_head_cfg['head_cls_name']
            ]))
            cur_num_class = len(rpn_head_cfg['head_cls_name'])
            
            
            cur_head = CenterSingleHead(
                input_channels, self.last_kernel_size, cur_num_class,
                self.code_size,
                reg_config=reg_config,
                head_label_indices=head_label_indices,
                separate_reg=True,
                use_multihead=True,
                use_direction_classifier=self.use_direction_classifier,
                use_iou_head=self.use_iou_head
            )
            rpn_heads.append(cur_head)
        self.rpn_heads = nn.ModuleList(rpn_heads)

    def get_target_assigner(self):
        target_assigner = CenterBasedTargetAssigner(
            class_names=self.class_names,
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            proposal_stride=self.proposal_stride,
            pc_area_scope=self.pc_area_scope,
            gaussian_overlap=self.assigner_cfg['gaussian_overlap'],
            min_radius=self.assigner_cfg['min_radius'],
            use_multihead=self.use_multihead,
            code_size=self.code_size
        )
        return target_assigner

    def assign_targets(self, gt_boxes):
        return self.target_assigner.assign_targets(gt_boxes)


    def forward(self, inputs, gt_boxes=None):
        if self.shared_conv is not None:
            inputs = self.shared_conv(inputs)
        
        outputs = []
        for rpn_head in self.rpn_heads:
            outputs.append(rpn_head(inputs))
        
        cls_preds = [output['cls_preds'] for output in outputs]
        box_preds = [output['box_preds'] for output in outputs]
        if 'iou_preds' in outputs[0]:
            iou_preds = [output['iou_preds'] for output in outputs]
            ret = dict(cls_preds=cls_preds, box_preds=box_preds, iou_preds=iou_preds)
        else:
            iou_preds = None
            # Assume separate_multihead=True in OpenPCDet
            ret = dict(cls_preds=cls_preds, box_preds=box_preds)
        targets = dict()
         
        
        if self.training and gt_boxes is not None:
            targets_dict = self.assign_targets(
                gt_boxes=gt_boxes
            )
            targets.update(targets_dict)
            return ret, targets

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                cls_preds=cls_preds, box_preds=box_preds
            )
            
            if isinstance(batch_cls_preds, list):
                multihead_label_mapping = []
                for idx in range(len(batch_cls_preds)):
                    multihead_label_mapping.append(
                        self.rpn_heads[idx].head_label_indices
                    )
            else:
                multihead_label_mapping = None
            ret = dict(
                batch_cls_preds=batch_cls_preds,
                batch_box_preds=batch_box_preds,
                batch_iou_preds=iou_preds,
                multihead_label_mapping=multihead_label_mapping
            )
            return ret, targets

    
    def get_cls_layer_loss(self, inputs, targets):
        cls_preds = inputs['cls_preds']
        # heatmaps
        box_cls_labels = targets['box_cls_labels']
        box_cls_mask = targets['box_cls_mask']

        c_idx = 0
        cls_losses = 0
        
        for idx, cls_pred in enumerate(cls_preds):
            batch_size = cls_pred.shape[0]
            cur_num_class = self.rpn_heads[idx].num_class
            cls_pred = cls_pred.view(batch_size, -1, cur_num_class)
            
            #cls_pred = torch.clamp(cls_pred.sigmoid_(), min=1e-4, max=1-1e-4)
            # assume separate heads
            cls_target = box_cls_labels[
                :, :, c_idx:c_idx+cur_num_class
            ]
            cls_mask = box_cls_mask[
                :, :, c_idx:c_idx+cur_num_class
            ]
            #print(torch.sum(cls_mask * cls_target))
            # Note: current focal loss impl. not aligned with CenterPoint official impl.
            #weights = torch.ones_like(cls_pred) / torch.clamp(cls_mask.sum(), 1)
            hm_loss = self.cls_loss_func(cls_pred, cls_target, cls_mask).sum() 

            cls_losses += hm_loss

            c_idx += cur_num_class

        cls_losses *= self.cls_weight
        return cls_losses


    
    def get_box_reg_layer_loss(self, inputs, targets):
        box_preds = inputs['box_preds']
        box_reg_targets = targets['box_reg_targets']
        box_reg_mask = targets['box_reg_mask']
        box_cls_mask = targets['box_cls_mask']

        start_idx = c_idx = 0
        reg_losses = 0

        for idx, box_pred in enumerate(box_preds):
            batch_size = box_pred.shape[0]
            cur_num_class = self.rpn_heads[idx].num_class
            box_pred = box_pred.view(
                batch_size, -1, box_pred.shape[-1] 
            )
            #box_reg_target = box_reg_targets[
            #    :, :, c_idx:c_idx+cur_num_class
            #].sum(-2)
            #reg_mask = box_reg_mask[
            #    :, :, c_idx:c_idx+cur_num_class
            #].sum(-1).reshape(box_reg_mask.size(0), -1, 1)
            
            if len(box_preds) > 1:
                box_reg_target = box_reg_targets.squeeze(-2) * (box_cls_mask[
                    :, :, c_idx:c_idx+cur_num_class].sum(-1, keepdims=True) != 0)
                reg_mask = box_reg_mask * (box_cls_mask[
                    :, :, c_idx:c_idx+cur_num_class].sum(-1, keepdims=True) != 0)
            else:
                box_reg_target = box_reg_targets.squeeze(-2)
                reg_mask = box_reg_mask.squeeze(-1).reshape(box_reg_mask.size(0), -1, 1)

            reg_loss_src = self.reg_loss_func(
                box_pred, box_reg_target, reg_mask
            )
            reg_losses += reg_loss_src

            start_idx += box_pred.shape[1]
            c_idx += cur_num_class

        reg_losses *= self.reg_weight
        return reg_losses


    def get_box_iou_loss(self, inputs, targets):
        cls_preds = inputs['cls_preds']
        box_preds = inputs['box_preds']
        iou_preds = inputs['iou_preds']
        if isinstance(cls_preds, list):
            batch_size = cls_preds[0].shape[0]
            if len(cls_preds[0].shape) > 3:
                # H x W
                num_anchors = cls_preds[0].shape[1] * cls_preds[0].shape[2]
            else:
                num_anchors = cls_preds[0].shape[1]
        else:
            batch_size = cls_preds.shape[0]
            if len(cls_preds.shape) > 3:
                # H x W
                num_anchors = cls_preds.shape[1] * cls_preds.shape[2]
            else:
                num_anchors = cls_preds.shape[1]


        box_cls_labels = targets['box_cls_labels']
        box_reg_targets = targets['box_reg_targets']
        box_reg_mask = targets['box_reg_mask'].squeeze(-1)
        # batch_size, HxW, box_dim
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            cls_preds=cls_preds, box_preds=box_preds
        )
        _, batch_box_preds_gt = self.generate_predicted_boxes(
            cls_preds=box_cls_labels, box_preds=box_reg_targets
        )
        # num_boxes, 8
        batch_box_preds_keep = batch_box_preds[box_reg_mask == 1][:, :7]
        batch_box_preds_gt_keep = batch_box_preds_gt[box_reg_mask == 1][:, :7]
        batch_box_preds_keep = torch.cat(
            [batch_box_preds_keep, torch.zeros_like(batch_box_preds_keep[: , :1])], 1
        )
        batch_box_preds_gt_keep = torch.cat(
            [batch_box_preds_gt_keep, torch.zeros_like(batch_box_preds_gt_keep[: , :1])], 1
        )
        #batch_box_preds_keep = iou3d_utils.box_transform_for_iou_calculation(batch_box_preds_keep)
        #batch_box_preds_gt_keep = iou3d_utils.box_transform_for_iou_calculation(batch_box_preds_gt_keep)
        # iou: num_boxes
        iou_gt = iou3d_utils.correspond_boxes_iou3d_gpu(batch_box_preds_keep, batch_box_preds_gt_keep)
        #iou_gt = 2 * iou_gt - 1
        # iou_loss
        iou_preds = torch.cat(iou_preds, 1)
        iou_preds = iou_preds.reshape(batch_size, num_anchors)
        # num_boxes
        iou_preds_keep = iou_preds[box_reg_mask == 1]
        
        iou_loss = nn.functional.l1_loss(iou_preds_keep, iou_gt)
        return iou_loss



    def get_loss(self, inputs, targets):
        cls_loss = self.get_cls_layer_loss(inputs, targets)
        reg_loss = self.get_box_reg_layer_loss(inputs, targets)

        if 'iou_preds' not in inputs:
            return {
                'rpn_loss_cls': cls_loss,
                'rpn_loss_loc': reg_loss
            }
        else:
            iou_loss = self.get_box_iou_loss(inputs, targets)
            return {
                'rpn_loss_cls': cls_loss,
                'rpn_loss_loc': reg_loss,
                'rpn_loss_iou': iou_loss
            }


    def generate_predicted_boxes(self, cls_preds, box_preds):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)
        """
        if isinstance(cls_preds, list):
            batch_size = cls_preds[0].shape[0]
            if len(cls_preds[0].shape) > 3:
                # H x W
                num_anchors = cls_preds[0].shape[1] * cls_preds[0].shape[2]
            else:
                num_anchors = cls_preds[0].shape[1]
        else:
            batch_size = cls_preds.shape[0]
            if len(cls_preds.shape) > 3:
                # H x W
                num_anchors = cls_preds.shape[1] * cls_preds.shape[2]
            else:
                num_anchors = cls_preds.shape[1]


        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) \
            if not isinstance(box_preds, list) \
            else torch.cat([x.view(batch_size, num_anchors, -1) for x in box_preds], dim=1)

        loc = batch_box_preds[..., :3]
        
        # self.proposal_stride is a 3D tuple
        sh, sw = self.proposal_stride[0], self.proposal_stride[2]
        H = (self.pc_area_scope[0][1]-self.pc_area_scope[0][0]) / self.voxel_size[0] / sh
        W = (self.pc_area_scope[2][1]-self.pc_area_scope[2][0]) / self.voxel_size[2] / sw
        H, W = int(round(H)), int(round(W))
        xs, ys = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])

        # TBD: not multihead mode?
        if not isinstance(box_preds, torch.Tensor):
            xs = xs.view(1, 1, H, W).repeat(batch_size, len(box_preds), 1, 1).view(batch_size, -1).to(loc.device)
            ys = ys.view(1, 1, H, W).repeat(batch_size, len(box_preds), 1, 1).view(batch_size, -1).to(loc.device)
        else:
            xs = xs.view(1, H, W).repeat(batch_size, 1, 1).view(batch_size, -1).to(loc.device)
            ys = ys.view(1, H, W).repeat(batch_size, 1, 1).view(batch_size, -1).to(loc.device)

        xs = xs * sh * self.voxel_size[0] + self.pc_area_scope[0][0]
        ys = ys * sw * self.voxel_size[2] + self.pc_area_scope[2][0]
        loc[..., 0] = loc[..., 0] * sh * self.voxel_size[0] + xs
        loc[..., 2] = loc[..., 2] * sw * self.voxel_size[2] + ys

        sizes = torch.exp(batch_box_preds[..., 3:6])
        rot = torch.atan2(batch_box_preds[..., 6:7], batch_box_preds[..., 7:8])
        if self.code_size > 8:
            velo = batch_box_preds[..., 8:]
            batch_box_preds = torch.cat([loc, sizes, rot, velo], dim=-1)
        else:
            batch_box_preds = torch.cat([loc, sizes, rot], dim=-1)
        return batch_cls_preds, batch_box_preds
