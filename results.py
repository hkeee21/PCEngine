import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
import time
from torch.utils.cpp_extension import load
from torchsparse.utils.quantize import sparse_quantize
from utils import sparse_quantize, load_file, vanillaConv, CheckResults

hash_module = load(name="mapping_cuda",
                   sources=["backend/pybind_hash.cpp", 
                   "backend/hash.cu"],
                   verbose=True)

conv_module = load(name="conv_fwd_cuda",
                   sources=["backend/pybind_conv.cpp", 
                   "backend/spconv.cu"],
                   verbose=True)


if __name__ == '__main__': 
    device = torch.device('cuda')
    
    '''
    # To generate the inputs (COO + feats)
    # Here input size denotes the number of input nnz. 
    # Currently only odd kernel size is considered
    input_size, input_channel, kernel_size = 1000, 3, 3
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

    dev_coords = torch.tensor(coords, dtype=torch.int).to(device)
    dev_feats = torch.tensor(feats, dtype=torch.float).to(device)

    '''
    # real data test
    input_channel, kernel_size = 3, 3

    coord, colors, pcd = load_file("/home/hongke21/nfs/MinkowskiEngine/MinkowskiEngine/examples/1.ply")
    coord -= np.min(coord, axis=0, keepdims=True)
    voxel_size = 0.2
    coord, indices = sparse_quantize(coord, voxel_size, return_index=True)
    input_nnz = coord.shape[0]
    coords = coord
    feats = colors[indices]
    print("input nnz : %d" % input_nnz)

    dev_coords = torch.tensor(coords, dtype=torch.int).to(device)
    dev_feats = torch.tensor(feats, dtype=torch.float).to(device)

    # To generate the weights
    output_channel = 256
    weights = np.random.uniform(0, 1, size=(kernel_size ** 3, input_channel, output_channel))
    dev_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # map and output, for mem allocation observation
    dev_output = torch.zeros((input_nnz, output_channel), dtype=torch.float).to(device)
    dev_map = torch.zeros((input_nnz * (kernel_size ** 3)), dtype=torch.int).to(device)
    dev_knnz = torch.zeros((kernel_size ** 3), dtype=torch.int).to(device)
    
    dev_kidx = hash_module.mapping_cuda(
        dev_coords,
        kernel_size,
        dev_map,
        dev_knnz
    )

    with torch.no_grad(): 
        conv_module.conv_fwd_cuda(
            dev_coords,
            dev_feats,
            dev_weights,
            kernel_size,
            dev_map,
            dev_output,
            dev_knnz,
            dev_kidx
        )
    
    print('Sparse Conv Done.')

    vani_output = vanillaConv(
        nnz=input_nnz,
        c_in=input_channel,
        c_out=output_channel,
        in_c=coords,
        in_f=feats,
        kv=weights,
        ks=kernel_size
    )

    print('Vanilla Conv Done.')

    # results checking
    output_array = dev_output.clone().cpu().numpy()
    accu_error = CheckResults(
        len=input_nnz * output_channel,
        c_out=output_channel,
        results1=output_array,
        results2=vani_output
    )

    print('The accumulated abs error: %.4f' % accu_error)
    
    





