import numpy as np
import torch
import torch.nn as nn
from ..modules import ConvBlock
from .anchor_head_template import AnchorHeadTemplate
from typing import Union, List, Tuple, Optional

__all__ = ['AnchorHeadMulti', 'SingleHead']

# if 2 classes, 2-rotation anchor,
# cls should have 8 channels (2 cls * (2 x 2) anchors)
# if 1 class, 2-rotation anchor
# cls should have 2 channels (1 cls * (1 x 2) anchors)

class SingleHead(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 last_kernel_size: int,
                 num_class: int, 
                 code_size: int,
                 num_anchors_per_location: int,
                 reg_config: dict,
                 head_label_indices: Union[List[int], Tuple[int, ...], np.ndarray, torch.Tensor],
                 separate_reg: Optional[bool] = True,
                 use_multihead: Optional[bool] = True,
                 **kwargs):
        super().__init__()

        assert code_size is not None
        
        self.num_class = num_class
        self.code_size = code_size
        self.input_channels = input_channels
        self.last_kernel_size = last_kernel_size
        self.num_anchors_per_location = num_anchors_per_location
        self.separate_reg = separate_reg
        self.use_multihead = use_multihead
        self.reg_config = reg_config
        self.register_buffer(
            'head_label_indices', torch.tensor(head_label_indices).long()
        )
        mid_channels = self.reg_config['mid_channels']
        num_middle_layers = self.reg_config['num_middle_layers']
        self.num_dir_bins = kwargs.get('num_dir_bins', None)
        

        if not self.separate_reg:
            self.conv_cls = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.num_class,
                kernel_size=1
            )
            self.conv_box = nn.Conv2d(
                input_channels, self.num_anchors_per_location * code_size,
                kernel_size=1
            )
        else:
            code_size_cnt = 0
            self.conv_box = nn.ModuleDict()
            self.conv_box_names = []
            conv_cls_list = []
            c_in = input_channels
            for k in range(num_middle_layers):
                conv_cls_list += [
                    ConvBlock(
                        c_in, mid_channels, kernel_size=3,
                        stride=1, padding=1, bias=False
                    )
                ]
                c_in = mid_channels
            
            conv_cls_list += [
                nn.Conv2d(c_in, self.num_anchors_per_location * self.num_class, 
                    kernel_size=last_kernel_size, stride=1, 
                    padding=last_kernel_size // 2, bias=True
                )
            ]
            self.conv_cls = nn.Sequential(*conv_cls_list)
            
            for reg_name, reg_channel in self.reg_config['reg_list'].items():
                conv_list = []
                c_in = input_channels
                for k in range(num_middle_layers):
                    conv_list += [
                        ConvBlock(
                            c_in, mid_channels, kernel_size=3,
                            stride=1, padding=1, bias=False
                        )
                    ]
                    c_in = mid_channels
                conv_list += [
                    nn.Conv2d(c_in, self.num_anchors_per_location * int(reg_channel), 
                        kernel_size=last_kernel_size, stride=1, 
                        padding=last_kernel_size // 2, bias=True
                    )
                ]
                code_size_cnt += reg_channel
                self.conv_box[f'conv_{reg_name}'] = nn.Sequential(*conv_list)
                self.conv_box_names.append(f'conv_{reg_name}')
            
            assert code_size_cnt == code_size
                             
            

        if kwargs.get('use_direction_classifier', False):
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.num_dir_bins,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        if kwargs.get('use_iou_head', False):
            conv_list = []
            c_in = input_channels
            for k in range(num_middle_layers):
                conv_list += [
                    ConvBlock(
                        c_in, mid_channels, kernel_size=3,
                        stride=1, padding=1, bias=False
                    )
                ]
                c_in = mid_channels
            conv_list += [
                nn.Conv2d(c_in, 1, 
                    kernel_size=last_kernel_size, stride=1, 
                    padding=last_kernel_size // 2, bias=True
                )
            ]
            self.conv_iou = nn.Sequential(*conv_list)
        else:
            self.conv_iou = None

        self.init_weights()

    def init_weights(self):
        for m in self.conv_box.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
        pi = 0.01
        if isinstance(self.conv_cls, nn.Conv2d):
            nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        else:
            nn.init.constant_(self.conv_cls[-1].bias, -np.log((1 - pi) / pi))

    def forward(self, inputs, gt_boxes=None):
        outputs = dict()
        
        cls_preds = self.conv_cls(inputs)
        if not self.separate_reg:
            box_preds = self.conv_box(inputs)
        else:
            box_preds = []
            # the concat is problematic here! to be fixed.
            for reg_name in self.conv_box_names:
                cur_box_pred = self.conv_box[reg_name](inputs)
                # [B, #AnchorsPerLoc, code_size, H, W]                
                B, _, H, W = cur_box_pred.size()
                cur_box_pred = cur_box_pred.reshape(
                    B, self.num_anchors_per_location, -1, H, W
                )
                box_preds.append(cur_box_pred)
            box_preds = torch.cat(box_preds, dim=2)
        
        if not self.use_multihead:
            # [N, C, H, W] -> [N, H, W, C]
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  
        else:
            H, W = box_preds.shape[3:]
            B = box_preds.shape[0]
            #box_preds = box_preds.view(B, -1, self.code_size, H, W)
            cls_preds = cls_preds.view(B, -1, self.num_class, H, W)
            # [B, #AnchorsPerLoc, code_size, H, W] -> 
            # [B, #AnchorsPerLoc, H, W, code_size]
            box_preds = box_preds.permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.permute(0, 1, 3, 4, 2).contiguous()
            box_preds = box_preds.view(B, -1, self.code_size)
            cls_preds = cls_preds.view(B, -1, self.num_class)
        
        
        outputs.update(cls_preds=cls_preds)
        outputs.update(box_preds=box_preds)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(inputs)
            if not self.use_multihead:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            else:
                dir_cls_preds = dir_cls_preds.view(B, -1, self.num_dir_bins, H, W)
                # [B, #AnchorsPerLoc, num_dir_bins, H, W] ->
                # [B, #AnchorsPerLoc, H, W, num_dir_bins]
                dir_cls_preds = dir_cls_preds.permute(0, 1, 3, 4, 2).contiguous()
                dir_cls_preds = dir_cls_preds.reshape(B, -1, self.num_dir_bins)
            outputs.update(dir_cls_preds=dir_cls_preds)
        else:
            dir_cls_preds = None

        if self.conv_iou is not None:
            iou_preds = self.conv_iou(inputs)

            iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()
            iou_preds = iou_preds.reshape(B, -1)
            outputs.update(iou_preds=iou_preds)
        
        return outputs

class AnchorHeadMulti(AnchorHeadTemplate):
    def __init__(self, 
                 input_channels: int,
                 last_kernel_size: int,
                 num_class: int, 
                 class_names: List[str], 
                 grid_size: Union[List[float], Tuple[float, ...], np.ndarray], 
                 pc_area_scope: Union[List[List[float]], np.ndarray], 
                 negative_threshold: dict,
                 positive_threshold: dict,
                 mean_size: dict,
                 mean_center_z: dict,
                 reg_config: dict,
                 rpn_head_configs: dict,
                 **kwargs):
        super().__init__(
            num_class=num_class, class_names=class_names,
            grid_size=grid_size, pc_area_scope=pc_area_scope,
            negative_threshold=negative_threshold, positive_threshold=positive_threshold,
            mean_size=mean_size, mean_center_z=mean_center_z, use_multihead=True, **kwargs
        )
        # note that class_names order need to match the heads def.
        self.last_kernel_size = last_kernel_size
        self.reg_config = reg_config
        self.rpn_head_configs = rpn_head_configs
        
        
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
        self.rpn_heads = None
        self.make_multiheads(shared_channels)
        
    
    
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
            head_label_indices = [
                class_names.index(cur_name) + 1 \
                for cur_name in rpn_head_cfg['head_cls_name']
            ]
            cur_num_class = len(rpn_head_cfg['head_cls_name'])
            
            num_anchors_per_location = sum(
                [self.num_anchors_per_location[class_names.index(head_cls)]
                    for head_cls in rpn_head_cfg['head_cls_name']]
            )
            
            cur_head = SingleHead(
                input_channels, self.last_kernel_size, cur_num_class,
                self.box_coder.code_size, num_anchors_per_location,
                reg_config=reg_config,
                head_label_indices=head_label_indices,
                separate_reg=True,
                use_multihead=True,
                use_direction_classifier=self.use_direction_classifier,
                num_dir_bins=self.num_dir_bins
            )
            rpn_heads.append(cur_head)
        self.rpn_heads = nn.ModuleList(rpn_heads)
    
        
    def forward(self, inputs, gt_boxes=None):
        if self.shared_conv is not None:
            inputs = self.shared_conv(inputs)
        
        outputs = []
        for rpn_head in self.rpn_heads:
            outputs.append(rpn_head(inputs))
        
        cls_preds = [output['cls_preds'] for output in outputs]
        box_preds = [output['box_preds'] for output in outputs]
        
        # Assume separate_multihead=True in OpenPCDet
        ret = dict(cls_preds=cls_preds, box_preds=box_preds)
        targets = dict()
        if self.use_direction_classifier:
            dir_cls_preds = [output['dir_cls_preds'] for output in outputs]
        else:
            dir_cls_preds = None
        ret.update(dir_cls_preds=dir_cls_preds)        
        
        if self.training and gt_boxes is not None:
            targets_dict = self.assign_targets(
                gt_boxes=gt_boxes
            )
            targets.update(targets_dict)
            return ret, targets
        
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
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
                multihead_label_mapping=multihead_label_mapping
            )
            return ret, targets
    
    def get_cls_layer_loss(self, inputs, targets):
        pos_cls_weight = self.pos_cls_weight
        neg_cls_weight = self.neg_cls_weight
        
        
        cls_preds = inputs['cls_preds']
        box_cls_labels = targets['box_cls_labels']
        
        if not isinstance(cls_preds, list):
            cls_preds = [cls_preds]
        
        batch_size = int(cls_preds[0].shape[0])
        cared = box_cls_labels >= 0
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0 * neg_cls_weight
        positive_cls_weights = positives * 1.0 * pos_cls_weight
        cls_weights = (positive_cls_weights + negative_cls_weights).float()
        
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1
        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
                
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 
            self.num_class + 1, 
            dtype=cls_preds[0].dtype, 
            device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        start_idx = c_idx = 0
        cls_losses = 0
        
        for idx, cls_pred in enumerate(cls_preds):
            cur_num_class = self.rpn_heads[idx].num_class
            cls_pred = cls_pred.view(batch_size, -1, cur_num_class)
            # assume separate heads
            one_hot_target = one_hot_targets[
                :, start_idx:start_idx+cls_pred.shape[1], c_idx:c_idx+cur_num_class
            ]
            
            cls_weight = cls_weights[:, start_idx:start_idx+cls_pred.shape[1]]
            #print(torch.sum(torch.sigmoid(cls_pred.max(-1).values)>0.1))
            #print(torch.sigmoid(cls_pred.max(-1).values).max())
              
            cls_loss = self.cls_loss_func(
                cls_pred, one_hot_target, weights=cls_weight.unsqueeze(-1)
            )
            cls_loss = cls_loss.sum() / batch_size
            cls_loss *= self.cls_weight
            cls_losses += cls_loss
            start_idx += cls_pred.shape[1]
            c_idx += cur_num_class
        assert start_idx == one_hot_targets.shape[1]
        return cls_losses

    def get_box_reg_layer_loss(self, inputs, targets):
        box_preds = inputs['box_preds']
        box_dir_cls_preds = inputs.get('dir_cls_preds', None)
        box_reg_targets = targets['box_reg_targets']
        box_cls_labels = targets['box_cls_labels']
        
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        
        if not isinstance(box_preds, list):
            box_preds = [box_preds]
        
        batch_size = int(box_preds[0].shape[0])

        if isinstance(self.anchors, list):
            # assume use multihead
            # anchor shape: #cls * [H, W, [Z?], 1, #rot, 7]
            # after processing: #cls, #rot, H, W, [Z?], 7
            """
            anchors = torch.cat([
                anchor.permute(3, 4, 0, 1, 2, 5) for anchor in self.anchors
            ], 0)
            """
            anchors = torch.cat(self.anchors, -3)
            anchors = anchors.permute(2, 3, 0, 1, 4).contiguous()
            anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        else:
            anchors = self.anchors
        
        if box_dir_cls_preds is not None:
            if not isinstance(box_dir_cls_preds, list):
                box_dir_cls_preds = [box_dir_cls_preds]
            
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.dir_offset,
                num_bins=self.num_dir_bins
            )
        
        start_idx = 0
        reg_losses = 0
        dir_losses = 0
        
        for idx, box_pred in enumerate(box_preds):
            box_pred = box_pred.view(
                batch_size, -1, box_pred.shape[-1] 
            )
            box_reg_target = box_reg_targets[
                :, start_idx:start_idx + box_pred.shape[1]
            ]
            reg_weight = reg_weights[
                :, start_idx:start_idx + box_pred.shape[1]
            ]
            if box_dir_cls_preds is not None:
                box_pred_sin, reg_target_sin = self.add_sin_difference(
                    box_pred, box_reg_target
                )
                # [N, M]
                reg_loss_src = self.reg_loss_func(
                    box_pred_sin, reg_target_sin
                )
            else:
                reg_loss_src = self.reg_loss_func(
                    box_pred, box_reg_target
                )
            reg_loss_src *= reg_weight.unsqueeze(-1)
            reg_loss = reg_loss_src.sum() / batch_size
            reg_loss *= self.reg_weight
            #print(box_reg_target[reg_weight!=0], box_pred[reg_weight!=0])
            if box_dir_cls_preds is not None:
                box_dir_cls_pred = box_dir_cls_preds[idx]
                dir_logit = box_dir_cls_pred.view(batch_size, -1, self.num_dir_bins)
                dir_target = dir_targets[:, start_idx:start_idx + box_pred.shape[1]]
                dir_loss = self.dir_loss_func(
                    dir_logit.transpose(2, 1), dir_target)
                dir_loss *= reg_weight
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * self.dir_weight
            else:
                dir_loss = torch.zeros_like(reg_loss)
            
            reg_losses += reg_loss
            dir_losses += dir_loss
            
            start_idx += box_pred.shape[1]
        assert start_idx == box_reg_targets.shape[1]
        return reg_losses, dir_losses

