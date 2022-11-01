from turtle import circle
import numpy as np
import torch
from torch import nn
import time

from script.spconv import conv3d
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

    in_channel = 4
    mid_channel = 64
    out_channel = 96

    cinfo = dict()
    cinfo["in"] = [in_channel, mid_channel]
    cinfo["out"] = [mid_channel, out_channel]
    cinfo["kernel"] = [3] 

    warmup_conv = conv3d(in_channels=in_channel,
                out_channels=mid_channel,
                kernel_size=3, 
                tc_mode_16f=0).to(device)
    warmup_conv.eval()

    single_conv = conv3d(in_channels=mid_channel,
                out_channels=out_channel,
                kernel_size=3, 
                tc_mode_16f=0).to(device)
    single_conv.eval()

    # dataset = ModelNet40Dataset()
    dataset = KITTIDataset(
        path='/home/eva_data/hongke21/datasets/data_object_velodyne/data_object_velodyne/testing/velodyne/',
        size=99
    )
    '''dataset = S3DISDataset(
        data_root='/home/eva_data/hongke21/datasets/stanford_indoor3d',
        test_area=4,
        num_point=4096 * 8,
        block_size=2.0)'''
    DataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=sparse_collate_fn,
        shuffle=False)
    # TODO: get a max input nnz from Dataset
    sample_num = 4096 * 32
    buffer = build_conv_buffer(cinfo, sample_num, device)

    count = 0
    dur = 0
    all_nnz = 0
    with torch.no_grad():
        for i, batch in enumerate(DataLoader):
            if i == 6: break

            input = batch['input'].to(device)
            input_nnz = input.coords.size(0)
            print("input nnz: %d" % input_nnz)
            input.buffer = buffer
            output = warmup_conv(input)

            if i >= 5:
                # torch.cuda.synchronize()
                # start=time.time()
                torch.cuda.cudart().cudaProfilerStart()
                output = single_conv(output)
                torch.cuda.cudart().cudaProfilerStop()
                # torch.cuda.synchronize()
                # end=time.time()
                # dur += (end - start)
                # count += 1
                # all_nnz += input_nnz
    
    # inf_time = dur / count
    # ave_nnz = all_nnz / count

    # print("Batches: %d" % count)
    # print("Average nnz: %d" % ave_nnz)
    # print("Duration: {:.4f} ms".format(inf_time * 1000))
    

if __name__ == '__main__':
    main()
