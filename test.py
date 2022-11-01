import numpy as np
import torch
from typing import List, Tuple, Union, Dict
from itertools import repeat
import time
import argparse
from script.utils import sparse_quantize, load_file, build_conv_buffer
from script.spconv import conv3d
from script.sptensor import spTensor


'''
cuda_module = load(name="tag_profiling",
                   sources=["/home/hongke21/nfs/code/tag_profiling.cpp", "/home/hongke21/nfs/code/tag_profiling.cu"],
                   verbose=True)'''


if __name__ == '__main__': 

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=100000)
    parser.add_argument('--conv-layers', type=int, default=100)
    parser.add_argument('--in-channels', type=int, default=16)
    parser.add_argument('--out-channels', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--real-data', type=bool, default=False)
    args = parser.parse_args()

    iter_num = args.conv_layers
    
    if not args.real_data:
        # To generate the inputs (COO + feats)
        # Here input size denotes the number of input nnz. 
        # Currently only odd kernel size is considered
        input_size, input_channel, kernel_size, output_channel = args.input_size, args.in_channels, args.kernel_size, args.out_channels
        voxel_size = 0.1

        voxel_range = 4
        if input_size > 80000:
            voxel_range = 10
        # Currently only 3D inputs are considered
        coords = np.random.uniform(0, voxel_range, size=(input_size, 3))

        # voxelization
        coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
        input_nnz = coords.shape[0]
        print("input nnz: %d" % input_nnz)

        feats = np.random.uniform(0, 1, size=(input_nnz, input_channel)) 

        # Sort for random data
        # Real data is sorted already
        arrSortedIndex = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        coords = coords[arrSortedIndex]
        feats = feats[arrSortedIndex]

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats, dtype=torch.float)

    else:
        # real data test
        input_channel, kernel_size, output_channel = args.in_channels, args.kernel_size, args.out_channels

        coord, _, pcd = load_file("/home/hongke21/nfs/MinkowskiEngine/MinkowskiEngine/examples/1.ply")
        coord -= np.min(coord, axis=0, keepdims=True)
        voxel_size = 0.02
        coord, indices = sparse_quantize(coord, voxel_size, return_index=True)
        input_nnz = coord.shape[0]
        print("input nnz: %d" % input_nnz)
        feat = np.random.uniform(0, 1, size=(input_nnz, input_channel)) 
        coords = torch.tensor(coord, dtype=torch.int)
        feats = torch.tensor(feat, dtype=torch.float)
    
    input = spTensor(feats, coords).to(device)

    cinfo = dict()
    cinfo["in"] = [input_channel]
    cinfo["out"] = [output_channel]
    cinfo["kernel"] = [3, 5] 

    buffer = build_conv_buffer(cinfo, input_nnz, device)

    
    conv_warmup = conv3d(in_channels=input_channel,
                    out_channels=output_channel,
                    buffer=buffer,
                    kernel_size=5,
                    tc_mode_16f=0).to(device)

    conv1 = conv3d(in_channels=input_channel,
                out_channels=output_channel,
                buffer=buffer,
                kernel_size=kernel_size, 
                stride=2, 
                tc_mode_16f=0).to(device)
    
    conv2 = conv3d(in_channels=output_channel,
                out_channels=128,
                buffer=buffer,
                kernel_size=kernel_size, 
                stride=2, 
                tc_mode_16f=0).to(device)

    for _ in range(10):
        with torch.no_grad():

            _ = conv_warmup(input)
    

    # torch.cuda.synchronize()
    # start=time.time()
    # print("--------")

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iter_num):

        with torch.no_grad():

            output = conv1(input)
            output = conv2(output)

    
    torch.cuda.cudart().cudaProfilerStop()

    
    # torch.cuda.synchronize()
    # end=time.time()
    # inf_time=(end-start)/iter_num

    # print("Duration for a convolution operation : {:.4f} ms".format(inf_time * 1000))



    
    





