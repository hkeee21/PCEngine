import torch
import time
import argparse
from tqdm import tqdm

import spconv.pytorch as spconv
import spconv as spconv_core
from .S3DIS import S3DISDataset
from .ModelNet40 import ModelNet40Dataset
from .utils import sparse_collate_fn
from .KITTI import KITTIDataset

@torch.no_grad()
def spconv_kernel(c_mid: int, c_out: int, data: str, dataflow):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    spconv_core.constants.SPCONV_ALLOW_TF32 = True 

    c_in = 4

    warmup_conv = spconv.SubMConv3d(
                        in_channels=c_in,
                        out_channels=c_mid,
                        kernel_size=3,
                        indice_key='subm1').to(device)
    warmup_conv.eval()

    single_conv = spconv.SubMConv3d(
                        in_channels=c_mid,
                        out_channels=c_out,
                        kernel_size=3).to(device)

    model = spconv.SparseSequential(single_conv)
    model.eval()
    
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


    count = 0
    dur = 0
    i = 0
    with torch.no_grad():
        for batch in tqdm(DataLoader):
            # if i == 40: break

            input = batch['input']
            input_nnz = input.indices.size(0)
            # print('input nnz: %d' % input_nnz)
            outputs = warmup_conv(input)
            for j in range(10):
                outputs = warmup_conv(input)
            
            if i >= 5:
                torch.cuda.synchronize()
                start=time.time()
                _ = single_conv(outputs)
                torch.cuda.synchronize()
                end=time.time()
                dur += end - start
                count += 1
            
            i += 1
             

    inf_time = dur / count

    return inf_time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-channel', type=str, default='64')
    parser.add_argument('--out-channel', type=str, default='128')
    parser.add_argument('--dataset', type=str, default='s3dis')
    args = parser.parse_args()
    spconv_kernel(args.in_channel, args.out_channel, args.dataset, None)
