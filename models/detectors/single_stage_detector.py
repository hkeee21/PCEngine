import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.iou3d import iou3d_utils
from ..heads import AnchorHead, AnchorHeadMulti

from typing import Optional, Union, List, Tuple

import time

__all__ = ['SingleStageDetector', 'AnchorBasedSingleStageDetector']


class SingleStageDetector(nn.Module):
    def __init__(self,
                 classes: List[str],
                 voxel_size: Union[np.ndarray, List[float]],
                 pc_area_scope: Union[np.ndarray, List[List[float]]],
                 last_kernel_size: int,
                 num_dir_bins: int,
                 cls_weight: float,
                 reg_weight: float,
                 dir_weight: float,
                 sep_heads: Optional[bool] = False,
                 use_multihead: Optional[bool] = False,
                 reg_config: Optional[dict] = None,
                 rpn_head_configs: Optional[dict] = None,
                 backbone_2d_configs: Optional[dict] = None,
                 nms_configs: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.classes = classes
        self.num_dir_bins = num_dir_bins
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.dir_weight = dir_weight
        self.voxel_size = voxel_size
        self.pc_area_scope = np.array(pc_area_scope)

        self.loc_max = np.round((self.pc_area_scope[:, 1]-self.pc_area_scope[:, 0])/self.voxel_size)
        self.loc_min = np.zeros_like(self.loc_max)
        
        self.sep_heads = sep_heads
        self.use_multihead = use_multihead

        # To be implemented
        self.encoder = None
        self.to_bev = None
        self.rpn = None
        self.head = None
        
        H = (self.pc_area_scope[0][1]-self.pc_area_scope[0][0]) / self.voxel_size[0]
        W = (self.pc_area_scope[2][1]-self.pc_area_scope[2][0]) / self.voxel_size[2]
        self.H, self.W = int(round(H)), int(round(W))
    
        if nms_configs is None:
            self.nms_configs = dict(
                roi_threshold=0.1,
                nms_threshold=0.01,
                joint_nms=True,
                before_nms_max=4096,
                after_nms_max=500
            )
        else:
            self.nms_configs = nms_configs


    def gen_cfg_dict(self, lis):
        dic = dict()
        for idx, cls in enumerate(self.classes):
            dic[cls] = lis[idx]
        return dic

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def post_processing(self, 
                        inputs,
                        roi_threshold=0.1,
                        nms_threshold=0.01,
                        joint_nms=True,
                        before_nms_max=4096,
                        after_nms_max=500):
        """
        Args:
            inputs:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        # Note: not yet supported multiple heads
        # not yet supported different length batches
        
        if isinstance(inputs['batch_cls_preds'], list):
            batch_size = inputs['batch_cls_preds'][0].shape[0]
            batch_cls_preds = [torch.sigmoid(x) for x in inputs['batch_cls_preds']]
        else:
            batch_size = inputs['batch_cls_preds'].shape[0]
            batch_cls_preds = torch.sigmoid(inputs['batch_cls_preds'])
        
        batch_box_preds = inputs['batch_box_preds']
        if 'batch_iou_preds' in inputs and inputs['batch_iou_preds'] is not None:
            batch_iou_preds = [x.reshape(batch_size, -1) for x in inputs['batch_iou_preds']]
        multihead_label_mapping = inputs.get('multihead_label_mapping', None)
        
        roi_boxes_all = []
        roi_cls_all = []
        
        for index in range(batch_size):
            if isinstance(batch_cls_preds, list):
                cls_preds = [x[index] for x in batch_cls_preds]
            else:
                cls_preds = batch_cls_preds[index]
            
            if 'batch_iou_preds' in inputs and inputs['batch_iou_preds'] is not None:
                iou_preds = [x[index] for x in batch_iou_preds]
            box_preds = batch_box_preds[index]
            box_preds = torch.cat([
                box_preds,
                torch.zeros_like(box_preds[:, 0:1])
            ], -1)
                        
            if not self.use_multihead:
                # do joint NMS
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                selected = iou3d_utils.nms_gpu_with_threshold(
                    box_preds, cls_preds, roi_threshold, 
                    nms_threshold, before_nms_max, after_nms_max
                )
                final_scores = cls_preds[selected]
                final_boxes = box_preds[selected]
                final_labels = label_preds[selected]
                final_predictions = torch.cat([
                    final_scores.unsqueeze(-1),
                    final_boxes[:, :-1],
                    torch.zeros_like(final_boxes[:, 0:1]) + index,
                ], 1)
                roi_boxes_all.append(final_predictions)
                roi_cls_all.append(final_labels)                
            else:
                # do class separate NMS
                assert multihead_label_mapping is not None
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for head_idx, (cur_cls_preds, cur_label_mapping) in enumerate(zip(cls_preds, multihead_label_mapping)):
                    if 'batch_iou_preds' in inputs and inputs['batch_iou_preds'] is not None:
                        cur_iou_preds = iou_preds[head_idx][cur_start_idx:cur_start_idx+cur_cls_preds.shape[0]]
                    else:
                        cur_iou_preds = None
                    cur_box_preds = box_preds[cur_start_idx:cur_start_idx+cur_cls_preds.shape[0]]
                    if joint_nms:
                        pred_score, pred_label, pred_box = iou3d_utils.multi_class_nms_gpu_with_threshold(
                            cur_box_preds, cur_cls_preds, roi_threshold, 
                            nms_threshold, before_nms_max, after_nms_max,
                            ious=cur_iou_preds
                        )
                    else:
                        pred_score, pred_label = cur_cls_preds.max(1)
                        selected = iou3d_utils.nms_gpu_with_threshold(
                            cur_box_preds, pred_score, roi_threshold,
                            nms_threshold, before_nms_max, after_nms_max,
                            ious=cur_iou_preds
                        )
                        if cur_iou_preds is not None:
                            selected, pred_score = selected
                        else:
                            pred_score = pred_score[selected]
                        pred_label = pred_label[selected]
                        pred_box = cur_box_preds[selected]
                    cur_start_idx += cur_cls_preds.shape[0]
                    pred_label = pred_label.long()
                    pred_label = cur_label_mapping[pred_label] - 1
                    pred_labels.append(pred_label)
                    
                    pred_boxes.append(
                        torch.cat([
                            pred_score.unsqueeze(-1), pred_box,
                            torch.zeros_like(pred_box[:, 0:1]) + index
                        ], 1)
                    )
                roi_boxes_all.extend(pred_boxes)
                roi_cls_all.extend(pred_labels)
                
        roi_boxes_all = torch.cat(roi_boxes_all, 0)
        roi_cls_all = torch.cat(roi_cls_all, 0)
        roi_boxes_dict = dict([
            (c, roi_boxes_all[roi_cls_all == i])
            for i, c in enumerate(self.classes)
        ])
        
        return roi_boxes_dict
                

    def forward(self,
                inputs,
                roi_threshold=0.1, # score after sigmoid
                nms_threshold=0.01,
                gt_scoring=False,
                joint_nms=True):
        x = inputs['pts_input']
        
        # torch.cuda.synchronize()
        # st = time.time()

        features = self.encoder(x)[-1]

        # print(features.feats.shape)
        
        # torch.cuda.synchronize()
        # ed = time.time()
        # print('3D backbone', (ed-st) * 1000)
        # print('3D backbone', [(k, v[2].sum()) for k, v in x.kmaps.items()])#x.F.shape, [(k, v[1]) for k, v in x.kmaps.items()], [(k, v[0].shape[0]) for k, v in x.kmaps.items()])
        # print('3D backbone', [(k, torch.sum(v[3]).item()) for k, v in features.indice_dict.items()])
        
        # torch.cuda.synchronize()
        # st = time.time()

        if self.to_bev is not None:
            features = self.to_bev(features)
        
        # torch.cuda.synchronize()
        # ed = time.time()
        # print('toBEV', (ed-st) * 1000)
        
        if self.sep_heads:
            if isinstance(features, list):
                # not gonna support this setting for CenterPoint
                raise NotImplementedError
            else:
                rpn_output = dict(
                    [(c, rpn_head(features)) 
                    for rpn_head, c in zip(self.rpn, self.classes)]
                )
        else:
            # torch.cuda.synchronize()
            # st = time.time()
            
            rpn_output = self.rpn(features)
            
            # torch.cuda.synchronize()
            # ed = time.time()
            # print('RPN', (ed-st) * 1000)
        
        num_classes = len(self.classes)
        
        if self.training:
            gt_boxes_key = 'gt_boxes' if 'gt_boxes' in inputs else 'gt_boxes3d'
            outputs, targets = self.head(rpn_output, inputs[gt_boxes_key])
            losses = self.head.get_loss(outputs, targets)
            return_dict = dict()
            return_dict.update(losses)
            return_dict['loss'] = sum(losses.values())
            return return_dict
        else:
            outputs, _ = self.head(rpn_output)

            #torch.cuda.synchronize()
            #st = time.time()

            roi_boxes_dict = self.post_processing(outputs, **self.nms_configs)


            #torch.cuda.synchronize()
            #ed = time.time()
            #print('nms', ed-st)


            outputs = inputs
            outputs.update({
                'rpn_output': rpn_output,
                'roi_boxes': roi_boxes_dict
            })
        return outputs


class AnchorBasedSingleStageDetector(SingleStageDetector):
    def __init__(self,
                 classes: List[str],
                 voxel_size: Union[np.ndarray, List[float]],
                 pc_area_scope: Union[np.ndarray, List[List[float]]],
                 mean_center_z: dict,
                 mean_size: dict,
                 last_kernel_size: int,
                 num_dir_bins: int,
                 cls_weight: float,
                 reg_weight: float,
                 dir_weight: float,
                 pos_cls_weight: float,
                 neg_cls_weight: float,
                 proposal_stage: int,
                 negative_anchor_threshold: List[float],
                 positive_anchor_threshold: List[float], 
                 max_number_of_voxels: Optional[Union[List[int], int]] = None,
                 max_points_per_voxel: Optional[Union[List[int], int]] = None,
                 sep_heads: Optional[bool] = False,
                 use_multihead: Optional[bool] = False,
                 reg_config: Optional[dict] = None,
                 rpn_head_configs: Optional[dict] = None,
                 backbone_2d_configs: Optional[dict] = None,
                 nms_configs: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        super().__init__(
            classes=classes,
            voxel_size=voxel_size,
            pc_area_scope=pc_area_scope,
            last_kernel_size=last_kernel_size,
            num_dir_bins=num_dir_bins,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            dir_weight=dir_weight,
            sep_heads=sep_heads,
            use_multihead=use_multihead,
            reg_config=reg_config,
            rpn_head_configs=rpn_head_configs,
            backbone_2d_configs=backbone_2d_configs,
            nms_configs=nms_configs,
            **kwargs
        )
        self.pos_cls_weight = pos_cls_weight
        self.neg_cls_weight = neg_cls_weight
        self.mean_center_z = mean_center_z
        self.mean_size = dict([(c,
                                torch.tensor(mean_size[c],
                                             dtype=torch.float32).cuda())
                               for c in self.classes])
        self.proposal_stage = proposal_stage
        if not use_multihead:
            self.rpn_head_configs = None
            self.reg_config = None
            head_class = AnchorHead
        else:
            self.rpn_head_configs = rpn_head_configs
            self.reg_config = reg_config
            
            assert self.rpn_head_configs is not None
            head_class = AnchorHeadMulti

        self.head = head_class(
            input_channels=sum(backbone_2d_configs['up_num_channels']),
            num_class=len(self.classes),
            use_direction_classifier=kwargs.get('use_direction_classifier', False),
            class_names=self.classes, 
            grid_size=[self.H, self.W], 
            pc_area_scope=self.pc_area_scope, 
            last_kernel_size=last_kernel_size,
            num_dir_bins=num_dir_bins,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            dir_weight=dir_weight,
            pos_cls_weight=pos_cls_weight,
            neg_cls_weight=neg_cls_weight,
            negative_threshold=self.gen_cfg_dict(negative_anchor_threshold),
            positive_threshold=self.gen_cfg_dict(positive_anchor_threshold),
            mean_size=self.mean_size,
            mean_center_z=self.mean_center_z,
            proposal_stage=self.proposal_stage,
            rpn_head_configs=self.rpn_head_configs,
            reg_config=self.reg_config,
            code_cfg=kwargs.get('code_cfg', None),
            loss_cfg=kwargs.get('loss_cfg', None)
        )


