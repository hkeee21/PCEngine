from math import ceil, floor
import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
import time
from torch.utils.cpp_extension import load
from script.utils import CheckResultsWeight, binary_search, sparse_quantize, \
    load_file, vanillaConv, CheckResults, vanillaConvBackward, \
    build_conv_buffer, sparse_collate_fn, conv_info_encoder
# from hashcublas.backend import mapping_cuda, conv_fwd_cuda
from PCEngine.backend import mapping_cuda, conv_fwd_cuda, mapping_simple_cuda, conv_fwd_simple_cuda
from configs.config import Config
from dataset_build import make_dataset
from model_build import make_model
import argparse

if __name__ == '__main__': 
    device = torch.device('cuda')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataflow', type=str, default='D1')
    args = parser.parse_args()

    save_name = 'v04-b1-k333-s111-i32-o32-FP32'
    data_type = torch.float
    batchsize = 1
    # real data test
    input_channel, kernel_size = 32, [3, 3, 3]
    output_channel = 32
    layer_stride = [1, 1, 1]
    kernel_size_code = conv_info_encoder(kernel_size)
    kernel_volume = np.prod(kernel_size, dtype=int)

    coord, _, pcd = load_file("data/1.ply")
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
    weights = np.random.uniform(0, 1, size=(kernel_volume, input_channel, output_channel))
    dev_weights = torch.tensor(weights, dtype=data_type).to(device)
    
    # map and output, for mem allocation observation
    dev_imap = - torch.ones((batchsize * input_nnz * kernel_volume), dtype=torch.int).to(device)
    dev_omap = - torch.ones(((batchsize * input_nnz * kernel_volume) * 2), dtype=torch.int).to(device)
    dev_knnz = torch.zeros((kernel_volume), dtype=torch.int).to(device)
    dev_kpos = torch.zeros((kernel_volume + 1), dtype=torch.int).to(device)
    dev_qkpos = torch.zeros((kernel_volume + 1), dtype=torch.int).to(device)
    dev_icsr = torch.zeros((batchsize * input_nnz + 2), dtype=torch.int).to(device)
    dev_ocsr = torch.zeros(((batchsize * input_nnz + 2) * 2), dtype=torch.int).to(device)

    layer_stride_code = conv_info_encoder(layer_stride)
    # make sure the input coordinates are at least tensor stride away from each other
    # TODO: can be further fused into the random data generator
    tensor_stride = [1, 1, 1]
    tensor_stride_code = conv_info_encoder(tensor_stride)

    separate_mid = (layer_stride[0] * layer_stride[1] * layer_stride[2]) == 1

    print(separate_mid)

    if args.dataflow == 'D2':

        dev_out_coords = mapping_simple_cuda(
            dev_coords, 
            batchsize, 
            kernel_size_code,
            kernel_volume,
            input_channel, 
            output_channel, 
            layer_stride_code,
            tensor_stride_code,
            dev_imap, 
            dev_knnz,
            dev_kpos, 
            dev_qkpos,
            separate_mid
        )

        qsum_nnz = dev_qkpos[-1].cpu().int()

        print("qsum nnz : %d" % qsum_nnz)

        nonzero_idx = torch.nonzero(dev_imap != -1)
        dev_omap = dev_imap[nonzero_idx]
        dev_imap = (nonzero_idx % (input_nnz * batchsize)).int()

        l = dev_out_coords.size(0)
        dev_output = torch.zeros((l, output_channel), dtype=data_type).to(device)

        print("output nnz : %d" % l)

        print("----- activated feature amount for each weight position -----")
        print(dev_knnz)
        print("----- accumulated feature address for each weight position -----")
        print(dev_kpos)
        print("----- quantified accumulated feature address for each weight position -----")
        print(dev_qkpos)

        with torch.no_grad(): 
            conv_fwd_simple_cuda(
                dev_feats,
                dev_weights,
                kernel_size_code, 
                qsum_nnz, 
                dev_output,
                dev_kpos, 
                dev_qkpos, 
                dev_imap,
                dev_omap,
                separate_mid, 
                0
            )
        
        print('Sparse Conv Done.')
    

    elif args.dataflow == 'D1':

        # forward validation

        dev_out_coords = mapping_cuda(
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
            dev_qkpos,
            separate_mid
        )

        # knnz = dev_knnz.cpu()
        qsum_nnz = dev_qkpos[-1].cpu().int()

        print("qsum nnz : %d" % qsum_nnz)

        print("----- activated feature amount for each weight position -----")
        print(dev_knnz)
        print("----- accumulated feature address for each weight position -----")
        print(dev_kpos)
        print("----- quantified accumulated feature address for each weight position -----")
        print(dev_qkpos)
        
        l = dev_out_coords.size(0)
        dev_output = torch.zeros((l, output_channel), dtype=data_type).to(device)

        print("output nnz : %d" % l)
        
        out_coords = dev_out_coords.clone().cpu().numpy()

        # dev_buf = torch.zeros((input_nnz * batchsize * 10 * (input_channel + output_channel), ), \
        #     dtype=data_type).to(device)

        cinfo = dict()
        cinfo["in"] = [input_channel]
        cinfo["out"] = [output_channel]
        cinfo["kernel"] = [2, 3] 
        dev_buf = build_conv_buffer(cinfo, 2 * input_nnz * batchsize, data_type, device)
        
        tensorcore_16F = 0

        # with torch.cuda.amp.autocast(enabled=enable_fp16):
        with torch.no_grad(): 
            conv_fwd_cuda(
                    dev_feats,
                    dev_weights,
                    kernel_size_code, 
                    qsum_nnz, 
                    dev_output,
                    dev_kpos, 
                    dev_qkpos,
                    dev_imap,
                    dev_omap,
                    dev_icsr, 
                    dev_ocsr, 
                    dev_buf, 
                    separate_mid, 
                    tensorcore_16F
                )
        
        print('Sparse Conv Done.')

    else:
        raise NotImplementedError
    
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

    
    '''
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







