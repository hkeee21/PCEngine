import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
from torch.utils.cpp_extension import load

backend = load(name="conv_fwd_cuda",
                   sources=["backend/pybind_cuda.cpp", 
                   "backend/spconv.cu"],
                   verbose=True)


if __name__ == '__main__': 
    device = torch.device('cuda')

    # To generate the inputs (COO + feats)
    # Here input size denotes the number of input nnz. 
    # Currently only input_channel==1 is considered
    input_size, voxel_size, input_channel, kernel_size = 10000, 0.1, 1, 3

    # Currently only 3D inputs are considered
    coords = np.random.randint(0, 100, size=(input_size, 3))
    feats = np.random.uniform(0, 1, size=(input_size, input_channel)) 

    # Sort for random data
    # Real data is sorted already
    arrSortedIndex = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    coords = coords[arrSortedIndex]
    feats = feats[arrSortedIndex]

    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(feats, dtype=torch.float)

    # To generate the weights
    # Currently only output_channel==1 is considered
    output_channel = 1
    weights = np.random.uniform(0, 1, size=(kernel_size ** 3, input_channel, output_channel))
    weights = torch.tensor(weights, dtype=torch.float)
    output = torch.zeros((input_size, input_channel))

    backend.conv_fwd_cuda(
                coords,
                feats,
                weights,
                kernel_size,
                output
            )
    
    





