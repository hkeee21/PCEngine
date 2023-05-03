import pandas as pd
import numpy as np
import argparse
import os
import sys
import torch
import time
from tqdm import tqdm
sys.path.append('../')
from lib.PCEngine.script.spconv import conv3d
from lib.PCEngine.script.sptensor import spTensor
from lib.PCEngine.script.utils import build_conv_buffer, load_file, sparse_quantize



@torch.no_grad()
def relative_kernel(c_mid: int, c_out: int, dataflow: str, nnz: int):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    c_in = 4

    cinfo = dict()
    cinfo["in"] = [c_in, c_mid]
    cinfo["out"] = [c_mid, c_out]
    cinfo["kernel"] = [3] 

    warmup_conv = conv3d(in_channels=c_in,
                out_channels=c_mid,
                kernel_size=3, 
                tc_mode=False).to(device)
    warmup_conv.eval()

    single_conv = conv3d(in_channels=c_mid,
                out_channels=c_out,
                kernel_size=3, 
                tc_mode=False).to(device)
    single_conv.eval()

    coord, _, pcd = load_file("/home/hongke21/nfs/MinkowskiEngine/MinkowskiEngine/examples/1.ply")
    coord -= np.min(coord, axis=0, keepdims=True)
    voxel_size = 0.4
    if nnz == 9971:
        voxel_size = 0.1
    elif nnz == 39147:
        voxel_size = 0.05
    elif nnz == 162055:
        voxel_size = 0.02
    
    coord, indices = sparse_quantize(coord, voxel_size, return_index=True)
    input_nnz = coord.shape[0]
    print("input nnz: %d" % input_nnz)
    coords = np.zeros((input_nnz, 4))
    coords[:, 0] = 0
    coords[:, 1:] = coord
    feat = np.random.uniform(0, 1, size=(input_nnz, c_in)) 
    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(feat, dtype=torch.float)
    coords_min = [0, 0, 0]
    coords_max = ((torch.max(coords[:, 1:], dim=0).values).cpu().numpy()).tolist()
    input = spTensor(feats=feats, coords=coords, \
                     coords_min=coords_min, coords_max=coords_max, buffer=None).to(device)
    
    # TODO: get a max input nnz from Dataset
    # sample_num = 4096 * 16
    buffer = build_conv_buffer(cinfo, input_nnz, torch.float, device)

    tag = ["end"]
    if dataflow == 'gather-mm-scatter':
        tag = ["whole", "whole", "end"]
    elif dataflow == 'fetch-on-demand':
        tag = ["simple", "simple", "end"]
    else:
        raise NotImplementedError

    count = 100
    dur = 0
    with torch.no_grad():
        for _ in range(count):
            # if i == 40: break

            input.buffer = buffer
            input.init_tag = tag
            for _ in range(10):
                output = warmup_conv(input)
                input.init_tag = tag
                # output.cbook.clear()
                # output.kmaps.clear()
            
            torch.cuda.synchronize()
            start=time.time()
            _ = single_conv(output)
            torch.cuda.synchronize()
            end=time.time()
            dur += (end - start)

    
    inf_time = dur / count

    return inf_time * 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-file', type=str, default='Fig7-relative')
    args = parser.parse_args()

    label_list = ['input channel', 'output channel', 'input nnz', 'dataflow', 'latency [ms]']
    channel_size_list = [(4, 16), (16, 16), (32, 32), (64, 64), (64, 96), (128, 128), (256, 256)]
    nnz_list = [9971, 39147, 162055]
    dataflow_list = ['gather-mm-scatter', 'fetch-on-demand']

    latency_array = np.zeros((len(channel_size_list) * len(nnz_list), len(dataflow_list)))
    c_in_col = []
    c_out_col = []
    nnz_col = []
    dataflow_col = []

    results_row = 0
    for _, channel_size in tqdm(enumerate(channel_size_list)):
        for nnz in nnz_list:
            results_col = 0
            for dataflow in dataflow_list:
                latency = relative_kernel(channel_size[0], channel_size[1], dataflow, nnz)
                c_in_col.append(channel_size[0])
                c_out_col.append(channel_size[1])
                nnz_col.append(nnz)
                dataflow_col.append(dataflow)
                latency_array[results_row, results_col] = latency
                results_col += 1
            results_row += 1

    latency_col = latency_array.reshape(-1).tolist()

    results = list(zip(c_in_col, c_out_col, nnz_col, dataflow_col, latency_col))
    results_csv = pd.DataFrame(data=results, columns=label_list)
    results_csv.to_csv(os.path.join('results', args.save_file + '.csv'), index=True, float_format='%.4f')

