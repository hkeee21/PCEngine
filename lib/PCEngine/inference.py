from turtle import circle
import numpy as np
import torch
from torch import nn
import time
import yaml
import argparse
from tqdm import tqdm
import os

from .models.minkunet import SparseResUNet42
from .models.resnet import SparseResNet21D
from .script.utils import build_conv_buffer, sparse_collate_fn
from .datasets.ModelNet40 import ModelNet40Dataset
from .datasets.S3DIS import S3DISDataset
from .datasets.KITTI import KITTIDataset


@torch.no_grad()
def pcengine_exe(net: str, data: str, dataflow: str, flag: bool=False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if net == 'resnet':
        file_name = os.path.join('../lib/PCEngine/configs', dataflow, 'resnet.yaml')
        backbone = SparseResNet21D
    elif net == 'unet':
        file_name = os.path.join('../lib/PCEngine/configs', dataflow, 'unet.yaml')
        backbone = SparseResUNet42
    else:
        raise NotImplementedError
    f = open(file_name, 'r')
    init_layer_dict = yaml.load(f, Loader=yaml.FullLoader)
    init_layer_tag = init_layer_dict['dataflow_tag']
    
    # print(f'{backbone.__name__}:')
    model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
    # print(model)
    model = model.to(device).eval()

    if flag:
        for name, module in model.named_modules():
            if 'conv3d' in module.__class__.__name__.lower():
                module.heuristics_flag = 1

    if data == 'modelnet40':
        dataset = ModelNet40Dataset(
            item=["piano", "bathtub", "airplane", "chair", "person", "sofa", "radio"],
            root="../AE-datasets/ModelNet40"
        )
    elif data == 'kitti':
        dataset = KITTIDataset(
            path='../AE-datasets/KITTI',
            size=1000
        )
    elif data == 's3dis':
        dataset = S3DISDataset(
            data_root='../AE-datasets/stanford_indoor3d',
            test_area=4,
            num_point=4096 * 4,
            block_size=1.0
        )
    else:
        raise NotImplementedError
    
    DataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=4 if flag and data in ['modelnet40', 's3dis'] else 1, 
        collate_fn=sparse_collate_fn,
        shuffle=False)
    
    sample_num = 4096 * 16
    buffer = build_conv_buffer({"in": [256], "out": [256], "kernel": [3]}, sample_num, torch.float, device)

    count = 0
    dur = 0
    i = 0
    with torch.no_grad():
        for batch in tqdm(DataLoader):
            # if i == 40: break

            input = batch['input']
            input_nnz = input.coords.size(0)
            if input_nnz < 4000: continue
            input = input.to(device)

            if i <= 5:
                input.buffer = buffer
                input.init_tag = init_layer_tag
                outputs = model(input)

            if i > 5:

                input.buffer = buffer
                input.init_tag = init_layer_tag
                torch.cuda.synchronize()
                start=time.time()
                outputs = model(input)
                torch.cuda.synchronize()
                end=time.time()
                count += 1
                dur += (end - start)
            
            i += 1
    
    inf_time = dur / count
    return inf_time


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='s3dis')
    parser.add_argument('--dataflow', type=str, default='heuristics')
    args = parser.parse_args()
    pcengine_exe(args.backbone, args.dataset, args.dataflow, False)