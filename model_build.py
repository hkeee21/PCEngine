from typing import Callable

import numpy as np
import torch
import torch.optim
from torch import nn
from configs.config import Config
from dataset_build import make_dataset

def make_model(configs, dataset) -> nn.Module:
    if configs.model.name == 'minkunet':
        from models.segmentation_models import MinkUNet
        param_dict = dict()
        param_dict.update(in_channels=configs.model.get('in_channels', 4))
        param_dict.update(cr=configs.model.get('cr', 1.0))
        param_dict.update(num_classes=configs.dataset.num_classes)
        model = MinkUNet(**param_dict)
        return model
        
    elif configs.model.name == 'centerpoint':
        from models.detectors import CenterPoint
        param_dict = dict()
        param_dict.update(in_channels = configs.model.get('in_channels', 4))
        param_dict.update(classes = dataset.classes)
        param_dict.update(pc_area_scope = configs.model.get('pc_area_scope', dataset.pc_area_scope))
        param_dict.update(last_kernel_size = configs.model.get('last_kernel_size', 1))
        param_dict.update(num_dir_bins = configs.model.get('num_dir_bins', None))
        param_dict.update(cls_weight = configs.model.cls_weight)
        param_dict.update(reg_weight = configs.model.reg_weight)
        param_dict.update(dir_weight = configs.model.dir_weight)
        param_dict.update(use_multihead = configs.model.get('use_multihead', False))
        param_dict.update(use_direction_classifier = configs.model.get('use_direction_classifier', False))
        param_dict.update(use_iou_head = configs.model.get('use_iou_head', False))
        param_dict.update(code_cfg = configs.model.get('box_code_cfg', 
            {'code_size': 7, 'encode_angle_by_sincos': False}
        ))
        param_dict.update(loss_cfg = configs.model.get('loss_cfg', None))
        param_dict.update(nms_configs = configs.model.get('nms_configs', None))
        param_dict.update(reg_config = configs.model.get('reg_config', None))
        param_dict.update(rpn_head_configs = configs.model.get('rpn_head_configs', None))
        param_dict.update(backbone_2d_configs = configs.model.get('backbone_2d_configs', None))

        if 'voxelization_cfg' not in configs.model:
            param_dict.update(voxel_size = dataset.voxel_size)
            param_dict.update(max_number_of_voxels = None)
            param_dict.update(max_points_per_voxel = None)
        else:
            param_dict.update(voxel_size = np.array(configs.model.voxelization_cfg.voxel_size, 'float'))
            # pointpillars
            param_dict.update(input_voxel_size = dataset.voxel_size)
            param_dict.update(max_number_of_voxels = configs.model.voxelization_cfg.max_number_of_voxels)
            param_dict.update(max_points_per_voxel = configs.model.voxelization_cfg.max_points_per_voxel)
        
        model = CenterPoint(**param_dict)

        if 'bn' in configs.model:
            for name, module in model.named_modules():
                if hasattr(module, 'reset_bn_params'):
                    module.reset_bn_params(
                        momentum=configs.model.bn.momentum,
                        eps=configs.model.bn.eps
                    )
        
    else:
        raise NotImplementedError(configs.model.name)
    return model


if __name__ == '__main__': 

    configs = Config()

    '''
    ### NS-CenterPoint(10f) ###
    configs.load('configs/default/nuscenes/centerpoint_default.yaml', recursive=True)
    configs.update(['model.cr', '1.0', 'model.enable_fp16', 'True'])
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/nuscenes'
    dataset = make_dataset(configs)
    model = make_model(configs, dataset)
    print(model)
    
    ### WM-CenterPoint(3f) ###
    configs.load('configs/default/waymo/centerpoint_default.yaml', recursive=True)
    configs.update(['model.cr', '1.0', 'model.enable_fp16', 'True', 'dataset.max_sweeps', '3'])
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/waymo'
    dataset = make_dataset(configs)
    model = make_model(configs, dataset)
    print(model)

    ### WM-CenterPoint(1f) ###
    configs.load('configs/default/waymo/centerpoint_default.yaml', recursive=True)
    configs.update(['model.cr', '1.0', 'model.enable_fp16', 'True', 'dataset.max_sweeps', '1'])
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/waymo'
    dataset = make_dataset(configs)
    model = make_model(configs, dataset)
    print(model)

    ### NS-MinkUNet(3f) ###
    configs.load('configs/default/nuscenes_lidarseg/default.yaml', recursive=True)
    configs.update(['model.cr', '1.0', 'model.enable_fp16', 'True', 'dataset.max_sweeps', '3'])
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/nuscenes'
    dataset = make_dataset(configs)
    model = make_model(configs, dataset)
    print(model)

    '''
    ### NS-MinkUNet(1f) ###
    configs.load('configs/default/nuscenes_lidarseg/default.yaml', recursive=True)
    configs.update(['model.cr', '1.0', 'model.enable_fp16', 'True', 'dataset.max_sweeps', '1'])
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/nuscenes'
    dataset = make_dataset(configs)
    model = make_model(configs, dataset)
    print(model)

    '''
    ### SK-MinkUNet(1.0x) ###
    configs.load('configs/default/semantic_kitti/default.yaml', recursive=True)
    configs.update(['model.cr', '1.0', 'model.enable_fp16', 'True'])
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/semantic-kitti'
    dataset = make_dataset(configs)
    model = make_model(configs, dataset)
    print(model)

    ### SK-MinkUNet(0.5x) ###
    configs.load('configs/default/semantic_kitti/default.yaml', recursive=True)
    configs.update(['model.cr', '0.5', 'model.enable_fp16', 'True'])
    configs.dataset.root = '/home/nfs_data/yangshang19/smr/torchsparse-data/benchmarks/semantic-kitti'
    dataset = make_dataset(configs)
    model = make_model(configs, dataset)
    print(model)'''




    