''' PCEngine @ hkeee21
    This code is to check forwad path of the sparse convolutional results.
        PCEngine Dataflow: Gather-MM-Scatter, Fetch-on-Demand (Heuristic
            dataflow selects between the two dataflows).
        Standards: SpConv (v2.2.6) results.
        Types: Submanifold & downsampling convolution.
        Please install SpConv (https://github.com/traveller59/spconv) first and run:
            $ python3 check.py
'''

import numpy as np
import torch
import time
from script.spconv import conv3d
from script.sptensor import spTensor
from script.utils import load_file, sparse_quantize
from spconv.pytorch import SubMConv3d, SparseConv3d
import spconv
import argparse


def coordinate_hash(coords: torch.Tensor):
    hash = torch.empty((coords.shape[0]), dtype=torch.int)
    hash = coords[:, 1] * (2 ** 20) + coords[:, 2] * (2 ** 10) + coords[:, 3]
    return hash


if __name__ == '__main__': 
    device = torch.device('cuda')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataflow', type=str, default='FetchonDemand')
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--voxel-size', type=float, default=0.2)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--in-channel', type=int, default=64)
    parser.add_argument('--out-channel', type=int, default=64)
    parser.add_argument('--dtype', type=str, default='fp32')
    args = parser.parse_args()

    ### data preprocess
    data_type = torch.half if args.dtype == 'fp16' else torch.float
    batchsize = args.batch_size
    c_in, c_out = args.in_channel, args.out_channel
    kernel_size = args.kernel_size
    stride = args.stride
    kernel_volume = np.prod(kernel_size, dtype=int)

    coord, _, pcd = load_file("sample-data/1.ply")
    coord -= np.min(coord, axis=0, keepdims=True)
    voxel_size = args.voxel_size
    coords, indices = sparse_quantize(coord, voxel_size, return_index=True)
    input_nnz = coords.shape[0]
    print('input nnz: %d' % input_nnz)
    feats = np.random.uniform(0, 1, size=(input_nnz, c_in))
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

    ### PCEngine sparse tensor
    buffer = torch.empty(((c_in + c_out) * input_nnz * kernel_volume,),
                         dtype=data_type,
                         device=device)
    coords_min = [0, 0, 0]
    coords_max = ((torch.max(coords[:, 1:], dim=0).values).cpu().numpy()).tolist()
    if args.dataflow == 'GatherScatter':
        layer_tag = ['D1', 'D1', 'end']
    elif args.dataflow == 'FetchonDemand':
        layer_tag = ['D2', 'D2', 'end']
    else:
        layer_tag = None
    input_pcengine = spTensor(coords=dev_coords, 
                              feats=dev_feats, 
                              batchsize=batchsize, 
                              buffer=buffer, 
                              init_tag=layer_tag, 
                              coords_min=coords_min, 
                              coords_max=coords_max)

    ### SpConv sparse tensor
    shape = np.max(coords.numpy(), axis=0) + 1
    shape = shape[1:]
    input_spconv = spconv.pytorch.SparseConvTensor(
                features=dev_feats, 
                indices=dev_coords, 
                spatial_shape=shape, 
                batch_size=batchsize)
    
    ### convolution setup
    conv_pcengine = conv3d(c_in, c_out, kernel_size, stride).to(device)
    if stride == 1:
        conv_spconv = SubMConv3d(c_in, c_out, kernel_size, bias=False).to(device)
    else:
        conv_spconv = SparseConv3d(c_in, c_out, kernel_size, stride, bias=False).to(device)
    spconv.constants.SPCONV_ALLOW_TF32 = True 
    weight = torch.rand((kernel_size, kernel_size, kernel_size, c_in, c_out), 
                        dtype=data_type, device=device)
    co_weight = weight.reshape(-1, c_in, c_out).to(device)
    re_weight = weight.permute(4, 0, 1, 2, 3).contiguous().to(device)
    conv_pcengine.state_dict()['kernel'].copy_(co_weight)
    conv_spconv.state_dict()['weight'].copy_(re_weight)
    if data_type == torch.half:
        conv_pcengine.half()
        conv_spconv.half()

    with torch.cuda.amp.autocast(enabled=(data_type == torch.half)):
        output_pcengine = conv_pcengine(input_pcengine)
        output_spconv = conv_spconv(input_spconv)

    PCEngie_coords = output_pcengine.coords
    SpConv2_coords = output_spconv.indices

    assert PCEngie_coords.shape == SpConv2_coords.shape

    N = PCEngie_coords.shape[0]
       
    PCEngie_coords_hash = coordinate_hash(PCEngie_coords) 
    SpConv2_coords_hash = coordinate_hash(SpConv2_coords)

    _, PCEngie_i = torch.sort(PCEngie_coords_hash)
    _, SpConv2_i = torch.sort(SpConv2_coords_hash)

    PCEngie_coords = PCEngie_coords[PCEngie_i]
    SpConv2_coords = SpConv2_coords[SpConv2_i]

    PCEngie_feats = output_pcengine.feats[PCEngie_i]
    SpConv2_feats = output_spconv.features[SpConv2_i]

    rtol = 1e-02 if data_type == torch.half else 1e-03
    atol = 1e-04 if data_type == torch.half else 1e-06

    all_close = torch.allclose(PCEngie_feats, SpConv2_feats, rtol=rtol, atol=atol)

    coords_err = torch.sum(torch.abs(PCEngie_coords - SpConv2_coords))

    mean_abs_err = torch.sum(torch.abs(PCEngie_feats - SpConv2_feats)) / (N * c_out)

    print('Ci: %d, Co: %d, Ks: %d, stride: %d, PCEngine (dataflow %s), MAE: %.4f, Coords Err: %d, All Close: %s.'\
          % (c_in, c_out, kernel_size, stride, args.dataflow, mean_abs_err, coords_err, bool(all_close)))





    



