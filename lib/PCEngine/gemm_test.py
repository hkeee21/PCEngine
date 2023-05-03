from math import ceil, floor
import numpy as np
import torch
from typing import List, Tuple, Union
from itertools import repeat
import time
from tqdm import tqdm
import argparse
from .script.utils import sparse_collate_fn, conv_info_encoder
from PCEngine.backend import mapping_simple_cuda, conv_fwd_simple_cuda, \
    conv_fwd_batched_cuda, conv_fwd_separate_cuda
from .datasets.ModelNet40 import ModelNet40Dataset
from .datasets.S3DIS import S3DISDataset
from .datasets.KITTI import KITTIDataset


def gemm_test(scheme: str, data: str, profile: bool):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    input_channel, output_channel, kernel_size = 64, 64, [3, 3, 3]
    layer_stride, tensor_stride, padding = [1, 1, 1], [1, 1, 1], [0, 0, 0]

    kernel_volume = np.prod(kernel_size, dtype=int)
    kernel_size_code = conv_info_encoder(kernel_size)
    padding_code = conv_info_encoder(padding)
    layer_stride_code = conv_info_encoder(layer_stride)
    tensor_stride_code = conv_info_encoder(tensor_stride)

    bs = 1
    if data == 'modelnet40':
        dataset = ModelNet40Dataset(
            item=["piano", "bathtub", "airplane", "chair", "person", "sofa", "radio"],
            root="../AE-datasets/ModelNet40"
        )
    elif data == 'kitti':
        dataset = KITTIDataset(
            path='../AE-datasets/KITTI',
            size=1000
        )
    elif data == 's3dis':
        dataset = S3DISDataset(
            data_root='../AE-datasets/stanford_indoor3d',
            test_area=4,
            num_point=4096 * 4,
            block_size=1.0
        )
        if profile:
            bs = 4
    else:
        raise NotImplementedError
    
    DataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=bs, 
        collate_fn=sparse_collate_fn,
        shuffle=False)
    

    count = 0
    dur = 0
    i = 0
    with torch.no_grad():
        for batch in tqdm(DataLoader):
            # if i == 20: break

            input = batch['input']
            coords = input.coords
            input_nnz = coords.size(0)
            feats = torch.rand((input_nnz, input_channel), dtype=torch.float)

            coords_min = coords[:, 1:].min(dim=0).values.numpy().tolist()
            coords_max = ((coords[:, 1:].max(dim=0).values.numpy() + 2 * np.array(padding) 
                   - (np.array(kernel_size) - 1)) // np.array(layer_stride)).tolist()

            dev_coords = coords.to(device)
            dev_feats = feats.to(device)

            # To generate the weights
            weights = np.random.uniform(0, 1, size=(kernel_volume, input_channel, output_channel))
            dev_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
            # mapping and output, for mem allocation observation
            dev_imap = - torch.ones((input_nnz * kernel_volume), dtype=torch.int).to(device)
            dev_omap = - torch.ones(((input_nnz * kernel_volume) * 2), dtype=torch.int).to(device)
            dev_knnz = torch.zeros((kernel_volume), dtype=torch.int).to(device)
            dev_kpos = torch.zeros((kernel_volume + 1), dtype=torch.int).to(device)
            dev_qkpos = torch.zeros((kernel_volume + 1), dtype=torch.int).to(device)
    
            separate_mid = (layer_stride[0] * layer_stride[1] * layer_stride[2]) == 1

            dev_out_coords = mapping_simple_cuda(
                dev_coords, 1, kernel_size_code, kernel_volume,
                input_channel, output_channel, 
                layer_stride_code, tensor_stride_code, padding_code, 
                coords_min[0], coords_min[1], coords_min[2], 
                coords_max[0], coords_max[1], coords_max[2], 
                dev_imap, dev_knnz, dev_kpos, dev_qkpos, separate_mid
            )

            qsum_nnz = dev_qkpos[-1].cpu().int()

            # print("qsum nnz : %d" % qsum_nnz)

            nonzero_idx = torch.nonzero(dev_imap != -1)
            dev_omap = dev_imap[nonzero_idx]
            dev_imap = (nonzero_idx % (input_nnz)).int()

            knnz = dev_knnz.cpu()
            sum_nnz = knnz.sum().int()

            out_nnz = dev_out_coords.size(0)
            dev_output = torch.zeros((out_nnz, output_channel), dtype=torch.float).to(device)

            # print("output nnz : %d" % out_nnz)

            if not profile:
                if scheme == 'fused':

                    for _ in range(10):
                        conv_fwd_simple_cuda(
                            dev_feats, dev_weights, kernel_size_code, 
                            qsum_nnz, dev_output, dev_kpos, dev_qkpos, 
                            dev_imap, dev_omap, separate_mid, 0
                        )   
                    torch.cuda.synchronize()
                    start=time.time()
                    conv_fwd_simple_cuda(
                        dev_feats, dev_weights, kernel_size_code, 
                        qsum_nnz, dev_output, dev_kpos, dev_qkpos, 
                        dev_imap, dev_omap, separate_mid, 0
                    )   
                    torch.cuda.synchronize()
                    end=time.time()
                    count += 1
                    dur += (end - start)

                elif scheme == 'batched':

                    M, theta = 40000, 0.8

                    for _ in range(10):
                        conv_fwd_batched_cuda(
                            dev_feats, dev_weights, kernel_size_code, 
                            sum_nnz, dev_output, knnz, dev_kpos, 
                            dev_imap, dev_omap, separate_mid, M, theta
                        )
                    torch.cuda.synchronize()
                    start=time.time()
                    conv_fwd_batched_cuda(
                        dev_feats, dev_weights, kernel_size_code, 
                        sum_nnz, dev_output, knnz, dev_kpos, 
                        dev_imap, dev_omap, separate_mid, M, theta
                    )
                    torch.cuda.synchronize()
                    end=time.time()
                    count += 1
                    dur += (end - start)

                elif scheme == 'separate':

                    for _ in range(10):
                        conv_fwd_separate_cuda(
                            dev_feats, dev_weights, kernel_size_code, 
                            sum_nnz, dev_output, knnz, dev_kpos, 
                            dev_imap, dev_omap, separate_mid
                        )
                    torch.cuda.synchronize()
                    start=time.time()
                    conv_fwd_separate_cuda(
                        dev_feats, dev_weights, kernel_size_code, 
                        sum_nnz, dev_output, knnz, dev_kpos, 
                        dev_imap, dev_omap, separate_mid
                    )
                    torch.cuda.synchronize()
                    end=time.time()
                    count += 1
                    dur += (end - start)

                else:
                    raise NotImplementedError
            else:
                M, theta = 30000, 0.5
                for _ in range(10):
                    conv_fwd_simple_cuda(
                            dev_feats, dev_weights, kernel_size_code, 
                            qsum_nnz, dev_output, dev_kpos, dev_qkpos, 
                            dev_imap, dev_omap, separate_mid, 0
                    )   
                torch.cuda.cudart().cudaProfilerStart()
                conv_fwd_simple_cuda(
                        dev_feats, dev_weights, kernel_size_code, 
                        qsum_nnz, dev_output, dev_kpos, dev_qkpos, 
                        dev_imap, dev_omap, separate_mid, 0
                )

                conv_fwd_batched_cuda(
                            dev_feats, dev_weights, kernel_size_code, 
                            sum_nnz, dev_output, knnz, dev_kpos, 
                            dev_imap, dev_omap, separate_mid, M, theta
                )

                conv_fwd_separate_cuda(
                        dev_feats, dev_weights, kernel_size_code, 
                        sum_nnz, dev_output, knnz, dev_kpos, 
                        dev_imap, dev_omap, separate_mid
                )
                torch.cuda.cudart().cudaProfilerStop()
                break
                
            i += 1

    if not profile:
        inf_time = dur / count

        return inf_time


if __name__ == '__main__': 
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme', type=str, default='fused')
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--profile', type=bool, default=False)
    args = parser.parse_args()

