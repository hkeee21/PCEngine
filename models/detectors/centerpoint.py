import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..backbone_3d import SparseResNet
from ..backbone_2d import DenseRPNHead, ToBEVConvolutionBlock
from ..heads import CenterHeadMulti
from typing import Union, List, Tuple
from .single_stage_detector import SingleStageDetector



__all__ = ['CenterPoint']


class CenterPoint(SingleStageDetector):
    def __init__(self,
                 classes: List[str],
                 voxel_size: Union[np.ndarray, List[float]],
                 pc_area_scope: Union[np.ndarray, List[List[float]]],
                 last_kernel_size: int,
                 num_dir_bins: int,
                 cls_weight: float,
                 reg_weight: float,
                 dir_weight: float,
                 sep_heads: bool = False,
                 use_multihead: bool = False,
                 reg_config: dict = None,
                 rpn_head_configs: dict = None,
                 backbone_2d_configs: dict = None,
                 nms_configs: dict = None,
                 **kwargs
                 ) -> None:
        super().__init__(
        	classes=classes, voxel_size=voxel_size, pc_area_scope=pc_area_scope,
        	last_kernel_size=last_kernel_size, num_dir_bins=num_dir_bins,
        	cls_weight=cls_weight, reg_weight=reg_weight, dir_weight=dir_weight,
        	sep_heads=sep_heads, use_multihead=use_multihead, reg_config=reg_config,
        	rpn_head_configs=rpn_head_configs, backbone_2d_configs=backbone_2d_configs,
        	nms_configs=nms_configs, **kwargs
        )

        self.encoder = SparseResNet(
            in_channels=kwargs.get('in_channels', 4)
        )

        self.to_bev = ToBEVConvolutionBlock(
            self.encoder.out_channels,
            128, 
            self.loc_min,
            self.loc_max,
            proposal_stride=[8, 16, 8],
        )

        self.rpn = DenseRPNHead(
            classes=self.classes,
            in_channels=256,
            loc_min=None,
            loc_max=None,
            proposal_stride=None,
            need_tobev=False,
            sep_heads=sep_heads,
            **backbone_2d_configs
        )

        if not use_multihead:
            raise NotImplementedError
        else:
            self.rpn_head_configs = rpn_head_configs
            self.reg_config = reg_config
            
            assert self.rpn_head_configs is not None
            self.head = CenterHeadMulti(
                input_channels=512,
                num_class=len(self.classes),
                use_direction_classifier=kwargs.get('use_direction_classifier', False),
                use_iou_head=kwargs.get('use_iou_head', False),
                class_names=self.classes, 
                voxel_size=voxel_size,
                proposal_stride=[8, 16, 8],
                grid_size=dict([
                    (c, [self.H, self.W]) for c in self.classes
                ]), 
                pc_area_scope=self.pc_area_scope,
                last_kernel_size=last_kernel_size,
                num_dir_bins=num_dir_bins,
                cls_weight=cls_weight,
                reg_weight=reg_weight,
                dir_weight=dir_weight,
                rpn_head_configs=self.rpn_head_configs,
                reg_config=self.reg_config,
                code_cfg=kwargs.get('code_cfg', None),
                loss_cfg=kwargs.get('loss_cfg', None)
            )

