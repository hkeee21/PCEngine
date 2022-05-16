import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
import time
from torch.utils.cpp_extension import load
from torchsparse.utils.quantize import sparse_quantize
from utils import sparse_quantize, load_file

backend = load(name="conv_fwd_cuda",
                   sources=["backend/pybind_cuda.cpp", 
                   "backend/spconv.cu"],
                   verbose=True)

if __name__ == '__main__': 
    device = torch.device('cuda')

    iter_num = 100
    
    '''
    # To generate the inputs (COO + feats)
    # Here input size denotes the number of input nnz. 
    # Currently only odd kernel size is considered
    input_size, input_channel, kernel_size = 20000, 3, 3
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

    coords = torch.tensor(coords, dtype=torch.int).to(device)
    feats = torch.tensor(feats, dtype=torch.float).to(device)

    '''
    # real data test
    input_channel, kernel_size = 3, 3

    coord, colors, pcd = load_file("data/1.ply")
    coord -= np.min(coord, axis=0, keepdims=True)
    voxel_size = 0.02
    coord, indices = sparse_quantize(coord, voxel_size, return_index=True)
    input_nnz = coord.shape[0]
    coords = torch.tensor(coord, dtype=torch.int).to(device)
    feats = torch.tensor(colors[indices], dtype=torch.float).to(device)
    

    # To generate the weights
    output_channel = 64
    weights = np.random.uniform(0, 1, size=(kernel_size ** 3, input_channel, output_channel))
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    # map and output, for mem allocation observation
    output = torch.zeros((input_nnz, output_channel), dtype=torch.float).to(device)
    map = torch.zeros((input_nnz,), dtype=torch.int).to(device)

    for _ in range(3):
        with torch.no_grad(): 
            backend.conv_fwd_cuda(
                coords,
                feats,
                weights,
                kernel_size,
                map,
                output
        )
    
    torch.cuda.synchronize()
    start=time.time()

    for _ in range(iter_num):

        with torch.no_grad(): 
            backend.conv_fwd_cuda(
                coords,
                feats,
                weights,
                kernel_size,
                map,
                output
        )
    
    
    torch.cuda.synchronize()
    end=time.time()
    inf_time=(end-start)/iter_num

    print("Duration for a convolution operation : {:.4f} ms".format(inf_time * 1000))


    
    





