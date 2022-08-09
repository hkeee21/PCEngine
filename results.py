import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
import time
from torch.utils.cpp_extension import load
from torchsparse.utils.quantize import sparse_quantize
from utils import CheckResultsWeight, sparse_quantize, load_file, vanillaConv, CheckResults, vanillaConvBackward
# from hashcublas.backend import mapping_cuda, conv_fwd_cuda

hash_module = load(name="mapping_cuda",
                   sources=["backend/pybind_hash.cpp", 
                   "backend/hash.cu"],
                   verbose=True)


conv_module = load(name="conv_fwd_cuda",
                   sources=["backend/pybind_conv.cpp", 
                   "backend/spconv.cu"],
                   verbose=True)

conv_back_module = load(name="conv_bwd_cuda",
                    sources=["backend/pybind_conv_back.cpp",
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
    input_channel, kernel_size = 16, 3

    coord, _, pcd = load_file("/home/hongke21/nfs/MinkowskiEngine/MinkowskiEngine/examples/1.ply")
    coord -= np.min(coord, axis=0, keepdims=True)
    voxel_size = 0.4
    coords, indices = sparse_quantize(coord, voxel_size, return_index=True)
    input_nnz = coords.shape[0]
    print('input nnz: %d' % input_nnz)
    feats = np.random.uniform(0, 1, size=(input_nnz, input_channel))

    dev_coords = torch.tensor(coords, dtype=torch.int).to(device)
    dev_feats = torch.tensor(feats, dtype=torch.float).to(device)

    # To generate the weights
    output_channel = 32
    weights = np.random.uniform(0, 1, size=(kernel_size ** 3, input_channel, output_channel))
    dev_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # map and output, for mem allocation observation
    dev_output = torch.zeros((input_nnz, output_channel), dtype=torch.float).to(device)
    dev_imap = - torch.ones((input_nnz, (kernel_size ** 3 - 1)), dtype=torch.int).to(device)
    dev_omap = - torch.ones((input_nnz, (kernel_size ** 3 - 1)), dtype=torch.int).to(device)
    dev_knnz = torch.zeros((kernel_size ** 3 - 1), dtype=torch.int).to(device)
    
    # torch.cuda.cudart().cudaProfilerStart()

    # forward validation

    hash_module.mapping_cuda(
        dev_coords,
        kernel_size,
        dev_imap,
        dev_omap,  
        dev_knnz
    )

    dev_kpos = torch.zeros((kernel_size ** 3 - 1), dtype=torch.int).to(device)
    for k in range(kernel_size ** 3 - 2):
        dev_kpos[k + 1] = dev_kpos[k] + dev_knnz[k]

    sum_nnz = dev_knnz.sum()
    dev_gbuf = torch.zeros((sum_nnz, input_channel), dtype=torch.float).to(device)
    dev_sbuf = torch.zeros((sum_nnz, output_channel), dtype=torch.float).to(device)

    tensorcore_16F = 0

    with torch.no_grad(): 
        conv_module.conv_fwd_cuda(
            dev_feats,
            dev_weights,
            kernel_size,
            dev_output,
            dev_knnz,
            dev_kpos, 
            dev_imap,
            dev_omap,
            dev_gbuf, 
            dev_sbuf, 
            tensorcore_16F
        )
    
    print('Sparse Conv Done.')

    # torch.cuda.cudart().cudaProfilerStop()
    '''
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
    
    
    in_map = dev_imap.clone().detach().cpu().numpy()
    out_map = dev_omap.clone().detach().cpu().numpy()

    print("input map")
    for i in range(input_nnz):
        for k in range((kernel_size ** 3 - 1)):
            # if (in_map[i, k] == -1):
            #     break
            print("%d - %d - %d" % (in_map[i, k], in_map[i, k] / 1186111, in_map[i, k] % 1186111))
        
        print("----------")
    '''

    # backward validation
    output_grad = np.random.uniform(0, 1, size=(input_nnz, output_channel))

    dev_output_grad = torch.tensor(output_grad, dtype=torch.float).to(device)
    dev_input_grad = torch.zeros((input_nnz, input_channel), dtype=torch.float).to(device)
    dev_weight_grad = torch.zeros((kernel_size ** 3, input_channel, output_channel), dtype=torch.float).to(device)

    conv_back_module.conv_bwd_cuda(
        dev_output_grad, 
        dev_feats,
        dev_weights,
        kernel_size,
        dev_input_grad, 
        dev_weight_grad, 
        dev_knnz,
        dev_kpos, 
        dev_imap,
        dev_omap,
        dev_gbuf, 
        dev_sbuf,  
        0
    )

    print('Back Propagation Done.')

    input_grad, weights_grad = vanillaConvBackward(
        nnz=input_nnz,
        c_in=input_channel, 
        c_out=output_channel, 
        in_c=coords, 
        in_f=feats,
        kw=weights, 
        ks=kernel_size,
        out_f_g=output_grad
    )

    input_grad_cuda = dev_input_grad.clone().cpu().numpy()
    weight_grad_cuda = dev_weight_grad.clone().cpu().numpy()

    in_grad_error = CheckResults(
        len=input_nnz * input_channel,
        c_out=input_channel,
        results1=input_grad_cuda,
        results2=input_grad
    )

    weights_grad_error = CheckResultsWeight(
        k_vol=kernel_size ** 3,
        c_in=input_channel, 
        c_out=output_channel,
        results1=weight_grad_cuda, 
        results2=weights_grad
    )


    print('The accumulated abs error of input gradients: %.4f' % in_grad_error)
    print('The accumulated abs error of weights gradients: %.4f' % weights_grad_error)
    







