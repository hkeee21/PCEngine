from turtle import circle
import numpy as np
import torch
from torch import nn
import time

from script.sptensor import spTensor
from models.minkunet import SparseResUNet42
from models.resnet import SparseResNet21D
from script.utils import build_conv_buffer, sparse_collate_fn
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.S3DIS import S3DISDataset
from datasets.KITTI import KITTIDataset


@torch.no_grad()
def main() -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cinfo = dict()
    cinfo["in"] = [256]
    cinfo["out"] = [256]
    cinfo["kernel"] = [2, 3] 

    # backbone = SparseResUNet42
    backbone = SparseResNet21D
    print(f'{backbone.__name__}:')
    model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
    # print(model)
    model = model.to(device).eval()

    dataset = S3DISDataset(
        data_root='/home/eva_data/hongke21/datasets/stanford_indoor3d',
        test_area=4,
        num_point=4096 * 8,
        block_size=2.0)
    '''dataset = KITTIDataset(
        path='/home/eva_data/hongke21/datasets/data_object_velodyne/data_object_velodyne/testing/velodyne/',
        size=50
    )'''
    # dataset = ModelNet40Dataset()
    DataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=sparse_collate_fn,
        shuffle=False)
    # TODO: get a max input nnz from Dataset
    sample_num = 4096 * 16
    buffer = build_conv_buffer(cinfo, sample_num, device)

    count = 0
    dur = 0
    with torch.no_grad():
        for i, batch in enumerate(DataLoader):
            if i == 7: break

            input = batch['input']
            input_nnz = input.coords.size(0)
            print("input nnz: %d" % input_nnz)
            input = input.to(device)

            if i <= 5:
                input.buffer = buffer
                outputs = model(input)

            if i > 5:
                # torch.cuda.synchronize()
                # start=time.time()
                torch.cuda.cudart().cudaProfilerStart()
                input.buffer = buffer
                outputs = model(input)
                torch.cuda.cudart().cudaProfilerStop()
                # torch.cuda.synchronize()
                # end=time.time()
                # count += 1
                # dur += (end - start)
    
    # inf_time = dur / count

    # print("Batches: %d" % count)
    # print("Duration: {:.4f} ms".format(inf_time * 1000))
    
    # for k, output in enumerate(outputs):
    #     print(f'output[{k}].F.shape = {output.feats.shape}')


if __name__ == '__main__':
    main()
