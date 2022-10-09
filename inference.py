from turtle import circle
import numpy as np
import torch
from torch import nn
import time

from script.sptensor import spTensor
from models.minkunet import SparseResUNet42
from models.resnet import SparseResNet21D
from script.utils import build_conv_buffer, sparse_collate_fn
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.S3DIS import S3DISDataset


@torch.no_grad()
def main() -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cinfo = dict()
    cinfo["in"] = [256]
    cinfo["out"] = [256]
    cinfo["kernel"] = [2, 3] 

    backbone = SparseResUNet42
    print(f'{backbone.__name__}:')
    model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
    print(model)
    model = model.to(device).eval()

    dataset = S3DISDataset(
        data_root='/home/nfs_data/hongke21/temp_dataset',
        test_area=2,
        num_point=4096 * 16,
        block_size=2.0)
    DataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=sparse_collate_fn,
        shuffle=False)
    # TODO: get a max input nnz from Dataset
    buffer = build_conv_buffer(cinfo, 100000, device)

    count = 0
    with torch.no_grad():
        for i, batch in enumerate(DataLoader):

            input = batch['input']
            input_nnz = input.coords.size(0)

            if i == 5:
                torch.cuda.synchronize()
                start=time.time()

            input.buffer = buffer
            input = input.to(device)

            outputs = model(input)

            count += 1

        
    torch.cuda.synchronize()
    end=time.time()
    inf_time=(end-start) / (count-5)

    print("Batched: %d" % i)
    print("Duration: {:.4f} ms".format(inf_time * 1000))
    
    for k, output in enumerate(outputs):
        print(f'output[{k}].F.shape = {output.feats.shape}')


if __name__ == '__main__':
    main()
