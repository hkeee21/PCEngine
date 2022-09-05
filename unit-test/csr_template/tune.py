#!/usr/bin/env python
import numpy as np
import torch
from kernel_tuner import tune_kernel
from kernel_tuner.util import get_config_string
from collections import OrderedDict
import json
import argparse
import sys
sys.path.append("../..")
from utils import sparse_quantize, load_file
from dgCloud.backend import mapping_cuda

def tune(chns: int, ks: int, vs: float):

    device = torch.device('cuda')

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(2, 8)]
    tune_params["block_size_y"] = [2**i for i in range(0, 5)]
    tune_params["block_size_z"] = [2**i for i in range(0, 5)]

    coord, _, _ = load_file("/home/hongke21/nfs/MinkowskiEngine/MinkowskiEngine/examples/1.ply")
    coord -= np.min(coord, axis=0, keepdims=True)
    voxel_size = vs
    coords, _ = sparse_quantize(coord, voxel_size, return_index=True)
    input_nnz = coords.shape[0]
    # print('input nnz: %d' % input_nnz)
    feats = np.random.uniform(0, 1, size=(input_nnz, chns)).astype(np.float32)

    dev_coords = torch.tensor(coords, dtype=torch.int).to(device)
    dev_feats = torch.tensor(feats, dtype=torch.float).to(device)

    dev_imap = - torch.ones((input_nnz * (ks ** 3 - 1)), dtype=torch.int).to(device)
    dev_omap = - torch.ones((input_nnz * (ks ** 3 - 1)), dtype=torch.int).to(device)
    dev_knnz = torch.zeros((ks ** 3), dtype=torch.int).to(device)
    dev_kpos = torch.zeros((ks ** 3 - 1), dtype=torch.int).to(device)
    dev_icsr = torch.zeros((input_nnz + 2), dtype=torch.int).to(device)
    dev_ocsr = torch.zeros((input_nnz + 2), dtype=torch.int).to(device)

    problem_size = input_nnz

    # derive the mapping
    mapping_cuda(
            dev_coords,
            ks,
            dev_imap,
            dev_omap, 
            dev_icsr, 
            dev_ocsr,  
            dev_knnz,
            dev_kpos
        )

    sum_nnz = dev_icsr[input_nnz].cpu().numpy()
    # dev_buf = torch.zeros((sum_nnz, chns), dtype=torch.float).to(device)

    buf = np.zeros((sum_nnz, chns)).astype(np.float32)
    kpos = dev_kpos.clone().cpu().numpy().astype(np.int32)
    icsr = dev_icsr.clone().cpu().numpy().astype(np.int32)
    imap = dev_imap.clone().cpu().numpy().astype(np.int32)

    args = [np.int32(input_nnz), np.int32(ks ** 3), np.int32(sum_nnz), 
        kpos, np.int32(chns), feats, icsr, imap, buf]

    grid_div_x = ["block_size_z"]
    grid_div_y = []
    grid_div_z = []

    restrict = ["block_size_x * block_size_y * block_size_z <= 1024"]

    # prepare output verification with custom function
    # reference = [np.sum(x), None, None]
    # def verify_partial_reduce(cpu_result, gpu_result, atol=None):
    #     return np.isclose(cpu_result[0], np.sum(gpu_result[0]), atol=atol)

    #tune the first kernel
    res, _ = tune_kernel("gather_all_input_major_csr_template3", "tune.cu", problem_size,
        args, tune_params, grid_div_x=grid_div_x, grid_div_y=grid_div_y, grid_div_z=grid_div_z, 
        restrictions=restrict, verbose=False, lang="CUDA", iterations=40)

    best_config = min(res, key=lambda x:x['time'])

    print("Channels=%d" % chns)
    print("Best performing config: \n" + get_config_string(best_config))
    
    # with open("tune.json", 'w') as fp:
    #     json.dump(res, fp)

    return res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--voxel-size', type=float, default=0.02)
    args = parser.parse_args()

    chns_list = [16, 32, 64, 96, 128, 192, 256, 512]
    for c in chns_list:
        tune(c, args.kernel_size, args.voxel_size)



