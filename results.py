from math import ceil, floor
import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
import time
from torch.utils.cpp_extension import load
from script.utils import CheckResultsWeight, binary_search, sparse_quantize, \
    load_file, vanillaConv, CheckResults, vanillaConvBackward
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
    data_type = torch.half
    batchsize = 2
    # real data test
    input_channel, kernel_size = 16, [3, 2, 3]
    kernel_size_code = 311 * kernel_size[0] + 17 * kernel_size[1] + kernel_size[2] 
    kernel_volume = np.prod(kernel_size, dtype=int)

    coord, _, pcd = load_file("/home/hongke21/nfs/MinkowskiEngine/MinkowskiEngine/examples/1.ply")
    coord -= np.min(coord, axis=0, keepdims=True)
    voxel_size = 0.4
    coords, indices = sparse_quantize(coord, voxel_size, return_index=True)
    input_nnz = coords.shape[0]
    print('input nnz: %d' % input_nnz)
    feats = np.random.uniform(0, 1, size=(input_nnz, input_channel))
    coords = torch.as_tensor(coords, dtype=torch.int)
    feats = torch.as_tensor(feats, dtype=data_type)

    # we use batch index as the first dimension of coordinates
    bcoords, bfeats = [], []
    for b in range(batchsize):
        batch = torch.full((input_nnz, 1),
                           b, dtype=torch.int)
        bcoords.append(torch.cat((batch, coords), dim=1))
        bfeats.append(feats)
    
    coords = torch.cat(bcoords, dim=0)
    feats = torch.cat(bfeats, dim=0)

    dev_coords = coords.to(device)
    dev_feats = feats.to(device)
    
    # To generate the weights
    output_channel = 16
    weights = np.random.uniform(0, 1, size=(kernel_volume, input_channel, output_channel))
    dev_weights = torch.tensor(weights, dtype=data_type).to(device)
    
    # map and output, for mem allocation observation
    dev_imap = - torch.ones((batchsize * input_nnz * kernel_volume), dtype=torch.int).to(device)
    dev_omap = - torch.ones((batchsize * input_nnz * kernel_volume), dtype=torch.int).to(device)
    dev_knnz = torch.zeros((kernel_volume), dtype=torch.int).to(device)
    dev_kpos = torch.zeros((kernel_volume), dtype=torch.int).to(device)
    dev_icsr = torch.zeros((batchsize * input_nnz + 2), dtype=torch.int).to(device)
    dev_ocsr = torch.zeros((batchsize * input_nnz + 2), dtype=torch.int).to(device)

    layer_stride = [1, 1, 1]
    layer_stride_code = 311 * layer_stride[0] + 17 * layer_stride[1] + layer_stride[2]
    # make sure the input coordinates are at least tensor stride away from each other
    # TODO: can be further fused into the random data generator
    tensor_stride = [1, 1, 1]
    tensor_stride_code = 311 * tensor_stride[0] + 17 * tensor_stride[1] + tensor_stride[2]

    separate_mid = (layer_stride[0] * layer_stride[1] * layer_stride[2]) == 1

    print(separate_mid)

    
    # forward validation

    dev_out_coords = hash_module.mapping_cuda(
        dev_coords,
        kernel_size_code,
        kernel_volume, 
        input_channel, 
        output_channel, 
        layer_stride_code,
        tensor_stride_code, 
        dev_imap,
        dev_omap,
        dev_icsr, 
        dev_ocsr,   
        dev_knnz,
        dev_kpos, 
        separate_mid
    )

    knnz = dev_knnz.cpu()
    sum_nnz = knnz.sum().int()

    print("sum nnz : %d" % sum_nnz)
    
    l = dev_out_coords.size(0)
    dev_output = torch.zeros((l, output_channel), dtype=data_type).to(device)

    print("output nnz : %d" % l)
    
    out_coords = dev_out_coords.clone().cpu().numpy()

    '''
    for i in range(l):
        print("(%d, %d, %d)" % (out_coords[i, 0], out_coords[i, 1], out_coords[i, 2]))

    print("------------------------")

    print(dev_knnz)
    print(dev_kpos)
    
    

    in_map = dev_imap.clone().detach().cpu().numpy()
    out_map = dev_omap.clone().detach().cpu().numpy()

    print("input map")
    for i in range(input_nnz):
        k = dev_icsr[i]
        while(k < dev_icsr[i + 1]):
            # if (in_map[i, k] == -1):
            #     break
            print("%d - %d" % (in_map[k] / 1186111, in_map[k] % 1186111))
            k += 1
        
        print("----------")

    print("--------------------------------------------")

    print("output map")
    for i in range(l):
        k = dev_ocsr[i]
        while(k < dev_ocsr[i + 1]):
            # if (in_map[i, k] == -1):
            #     break
            print("%d - %d" % (out_map[k] / 1186111, out_map[k] % 1186111))
            k += 1
        
        print("----------")

    print("--------------------------------------------")
    print("%d - %d" % (dev_icsr[input_nnz], dev_ocsr[l]))

    
    
    # some testings on the workload balance idea
    icsr = dev_icsr.clone().cpu().numpy()

    
    print("input map")
    for i in range(input_nnz):
        print("%d" % (icsr[i + 1] - icsr[i]))

    _MPNS_PER_BLOCK = 48
    block_num = ceil(sum_nnz / _MPNS_PER_BLOCK) 
    for b in range(block_num):
        m_start = b * _MPNS_PER_BLOCK
        m_end = m_start + _MPNS_PER_BLOCK
        if (m_end > sum_nnz):
            m_end = sum_nnz
        id_start = binary_search(icsr, m_start, 0, input_nnz - 1)
        id_end = id_start + 1
        while(icsr[id_end] < m_end):
            id_end += 1
        if (m_end - icsr[id_end - 1] <= icsr[id_end] - m_end):
            id_end -= 1
        # id_end = binary_search(icsr, m_end, 0, input_nnz - 1)
        print("%d-%d %d-%d (%d)" % (id_start, id_end, icsr[id_start], icsr[id_end] - 1, icsr[id_end] - icsr[id_start]))
        # print("%d" % (icsr[id_end] - icsr[id_start]))
    
    '''

    dev_buf = torch.zeros((input_nnz * batchsize * 10 * (input_channel + output_channel), ), \
        dtype=data_type).to(device)
    
    tensorcore_16F = 0

    with torch.no_grad(): 
        conv_module.conv_fwd_cuda(
            dev_feats,
            dev_weights,
            kernel_size_code, 
            sum_nnz, 
            dev_output,
            knnz,
            dev_kpos, 
            dev_imap,
            dev_omap,
            dev_icsr, 
            dev_ocsr, 
            dev_buf, 
            separate_mid, 
            tensorcore_16F
        )
    
    print('Sparse Conv Done.')

    # torch.cuda.cudart().cudaProfilerStop()
    
    vani_output = vanillaConv(
        c_in=input_channel,
        c_out=output_channel,
        stride=layer_stride, 
        in_c=coords,
        in_f=feats,
        kv=weights,
        ks=kernel_size
    )

    print('Vanilla Conv Done.')

    
    # results checking
    output_array = dev_output.clone().cpu().numpy()
    accu_error = CheckResults(
        len=l * output_channel,
        c_out=output_channel,
        results1=output_array,
        results2=vani_output
    )

    print('The accumulated abs error: %.4f' % accu_error)
    

    print(dev_buf)

    '''
    
    # mapping check

    in_map = dev_imap.clone().detach().cpu().numpy()
    out_map = dev_omap.clone().detach().cpu().numpy()

    print("input map")
    for i in range(input_nnz):
        k = dev_icsr[i]
        while(k < dev_icsr[i + 1]):
            # if (in_map[i, k] == -1):
            #     break
            print("%d" % (in_map[k]))
            k += 1
        
        print("----------")
    
    
    # backward validation

    output_grad = np.random.uniform(0, 1, size=(l, output_channel))

    dev_output_grad = torch.tensor(output_grad, dtype=torch.float).to(device)
    dev_input_grad = torch.zeros((batchsize * input_nnz, input_channel), dtype=torch.float).to(device)
    dev_weight_grad = torch.zeros((kernel_volume, input_channel, output_channel), dtype=torch.float).to(device)

    conv_back_module.conv_bwd_cuda(
        dev_output_grad, 
        dev_feats,
        dev_weights,
        kernel_size_code,
        sum_nnz,
        dev_input_grad, 
        dev_weight_grad, 
        knnz,
        dev_kpos, 
        dev_imap,
        dev_omap,
        dev_icsr,
        dev_ocsr,
        dev_buf, 
        0
    )

    print('Back Propagation Done.')

    input_grad, weights_grad = vanillaConvBackward(
        innz=batchsize * input_nnz,
        onnz=l,
        c_in=input_channel, 
        c_out=output_channel, 
        in_c=coords, 
        out_c=out_coords,
        in_f=feats,
        kw=weights, 
        ks=kernel_size,
        out_f_g=output_grad
    )

    input_grad_cuda = dev_input_grad.clone().cpu().numpy()
    weight_grad_cuda = dev_weight_grad.clone().cpu().numpy()

    in_grad_error = CheckResults(
        len=batchsize * input_nnz * input_channel,
        c_out=input_channel,
        results1=input_grad_cuda,
        results2=input_grad
    )

    weights_grad_error = CheckResultsWeight(
        k_vol=kernel_volume,
        c_in=input_channel, 
        c_out=output_channel,
        results1=weight_grad_cuda, 
        results2=weights_grad
    )

    print('The accumulated abs error of input gradients: %.4f' % in_grad_error)
    print('The accumulated abs error of weights gradients: %.4f' % weights_grad_error)
    
    '''







