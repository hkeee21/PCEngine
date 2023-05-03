import torch
import time
import argparse
from tqdm import tqdm

from .script.spconv import conv3d
from .script.utils import build_conv_buffer, sparse_collate_fn
from .datasets.ModelNet40 import ModelNet40Dataset
from .datasets.S3DIS import S3DISDataset
from .datasets.KITTI import KITTIDataset


@torch.no_grad()
def pcengine_kernel(c_mid: int, c_out: int, data: str, dataflow: str):
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
    else:
        raise NotImplementedError
    
    DataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        collate_fn=sparse_collate_fn,
        shuffle=False)
    
    # TODO: get a max input nnz from Dataset
    sample_num = 4096 * 16
    buffer = build_conv_buffer(cinfo, sample_num, torch.float, device)

    tag = ["end"]
    if dataflow == 'gather-mm-scatter':
        tag = ["whole", "whole", "end"]
    elif dataflow == 'fetch-on-demand':
        tag = ["simple", "simple", "end"]
    else:
        raise NotImplementedError

    count = 0
    dur = 0
    i = 0
    with torch.no_grad():
        for batch in tqdm(DataLoader):
            # if i == 40: break

            input = batch['input'].to(device)
            input_nnz = input.coords.size(0)
            input.buffer = buffer
            input.init_tag = tag
            for j in range(10):
                output = warmup_conv(input)
                input.init_tag = tag
                output.cbook.clear()
                output.kmaps.clear()

            if i >= 5:
                torch.cuda.synchronize()
                start=time.time()
                _ = single_conv(output)
                torch.cuda.synchronize()
                end=time.time()
                dur += (end - start)
                count += 1
            
            i += 1
    
    inf_time = dur / count

    return inf_time
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-channel', type=str, default='64')
    parser.add_argument('--out-channel', type=str, default='128')
    parser.add_argument('--dataset', type=str, default='s3dis')
    parser.add_argument('--dataflow', type=str, default='heuristics')
    args = parser.parse_args()
    pcengine_kernel(args.in_channel, args.out_channel, args.dataset, args.dataflow)