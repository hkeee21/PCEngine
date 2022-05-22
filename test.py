import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
import time
import argparse
from utils import sparse_quantize, load_file, vanillaConv
# from spconvmod.backend import conv_fwd_cuda
from spconv import conv3d
from sptensor import spTensor


if __name__ == '__main__': 

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=100000)
    parser.add_argument('--conv-layers', type=int, default=10)
    parser.add_argument('--in-channels', type=int, default=32)
    parser.add_argument('--out-channels', type=int, default=64)
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

    
        # Currently only 3D inputs are considered
        coords = np.random.uniform(0, 10, size=(input_size, 3))

        # voxelization
        coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
        input_nnz = coords.shape[0]

        feats = np.random.uniform(0, 1, size=(input_nnz, input_channel)) 

        # Sort for random data
        # Real data is sorted already
        arrSortedIndex = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        coords = coords[arrSortedIndex]
        feats = feats[arrSortedIndex]

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats, dtype=torch.float)
        input = spTensor(feats, coords).to(device)

    else:
        # real data test
        input_channel, kernel_size, output_channel = 3, args.kernel_size, args.out_channels

        coord, colors, pcd = load_file("data/1.ply")
        coord -= np.min(coord, axis=0, keepdims=True)
        voxel_size = 0.02
        coord, indices = sparse_quantize(coord, voxel_size, return_index=True)
        input_nnz = coord.shape[0]
        coords = torch.tensor(coord, dtype=torch.int)
        feats = torch.tensor(colors[indices], dtype=torch.float)
        input = spTensor(feats, coords).to(device)
    

    conv_warmup = conv3d(in_channels=input_channel,
                        out_channels=output_channel,
                        kernel_size=5
    ).to(device)

    conv = conv3d(in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=kernel_size).to(device)

    for _ in range(3):
        with torch.no_grad(): 
            _ = conv_warmup(input)
    

    torch.cuda.synchronize()
    start=time.time()

    for _ in range(iter_num):
        # cuda_module.torch_launch_tag_profiling()

        with torch.no_grad(): 
            output = conv(input)

        # cuda_module.torch_launch_tag_profiling()
    
    
    torch.cuda.synchronize()
    end=time.time()
    inf_time=(end-start)/iter_num

    print("Duration for a convolution operation : {:.4f} ms".format(inf_time * 1000))



    
    





