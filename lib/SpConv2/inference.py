from random import shuffle
from turtle import circle
import numpy as np
import torch
from torch import nn
import time
import argparse
from tqdm import tqdm

import spconv as spconv_core
from .models import SparseResUNet42, SparseResNet21D
from .S3DIS import S3DISDataset
from .ModelNet40 import ModelNet40Dataset
from .utils import sparse_collate_fn
from .KITTI import KITTIDataset


@torch.no_grad()
def spconv_exe(net: str, data: str, dataflow: str):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    spconv_core.constants.SPCONV_ALLOW_TF32 = True 

    if net == 'resnet':
        backbone = SparseResNet21D
    elif net == 'unet':
        backbone = SparseResUNet42
    else:
        raise NotImplementedError
    # print(f'{backbone.__name__}:')
    model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
    # print(model)
    model = model.to(device).eval()
    
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
        batch_size=1, 
        collate_fn=sparse_collate_fn,
        shuffle=False)

    count = 0
    dur = 0
    i = 0
    with torch.no_grad():
        for batch in tqdm(DataLoader):
            # if i == 40: break

            input = batch['input']
            input_nnz = input.indices.size(0)
            if input_nnz < 4000: continue
          
            if i <= 5:
                outputs = model(input)

            if i > 5:
            
                torch.cuda.synchronize()
                start=time.time()
                outputs = model(input)
                torch.cuda.synchronize()
                end=time.time()
                dur += end - start
                count += 1
            
            i += 1
             

    inf_time = dur / count

    return inf_time


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--dataset', type=str, default='s3dis')
    parser.add_argument('--dataflow', type=str, default='heuristics')
    args = parser.parse_args()
    spconv_exe(args.backbone, args.dataset, args.dataflow)