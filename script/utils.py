from mimetypes import init
import open3d as o3d
from typing import List, Tuple, Union, Dict, Callable, Any
from itertools import repeat
import numpy as np
import torch
from math import ceil, floor
from .sptensor import spTensor


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs


def build_conv_buffer(channel_list: Dict,
                    nnz: int, 
                    data_type, 
                    device):

    max_c_in = max(channel_list["in"])
    max_c_out = max(channel_list["out"])
    max_ks = max(channel_list["kernel"])

    if max_ks == 3:
        buffer_size = (max_c_in + max_c_out) * nnz * 16
    elif max_ks == 5:
        buffer_size = (max_c_in + max_c_out) * nnz * 48
    elif max_ks == 2:
        buffer_size = (max_c_in + max_c_out) * nnz * 8
    else:
        buffer_size = (max_c_in + max_c_out) * nnz * (max_ks ** 3 - 1)

    # print(buffer_size)
    buffer = torch.zeros((buffer_size,), dtype=data_type, device=device)

    return buffer
    

def init_layer_scheduler(conv_info: Dict):

    stride = conv_info['stride']
    in_channel = conv_info['in']
    out_channel = conv_info['out']

    # if the channels are relatively small and 
    # the usage of fetch-on-demand method has little 
    # impact on the computation later
    init_layer_hashmap_type = 'simple'

    for i, s in enumerate(stride):
        if s > 1: break
        # if the channels are too large,
        # we should abandon the fetch-on-demand method
        if in_channel[i] > 32 or out_channel[i] > 32:
            init_layer_hashmap_type = 'whole'
            return init_layer_hashmap_type
    
    # if there exists transposed/inversed layer
    if any(stride) < 1:
        mul = 1
        # check if the channel of the layers that \
        # uses the same coords as the init layer is too large
        for i, s in enumerate(stride):
            mul *= s
            if mul == 1:
                if in_channel[i] > 32 or out_channel[i] > 32:
                    init_layer_hashmap_type = 'whole'
                    return init_layer_hashmap_type

    return init_layer_hashmap_type


def output_stride_compute(
    l_stride_code: int, 
    t_stride_code: int,
    transposed: bool = False
):
    l_stride_x = l_stride_code // 94273
    l_stride_y = (l_stride_code - l_stride_x * 94273) // 311 
    l_stride_z = (l_stride_code - l_stride_x * 94273 - l_stride_y * 311)

    t_stride_x = t_stride_code // 94273
    t_stride_y = (t_stride_code - t_stride_x * 94273) // 311 
    t_stride_z = (t_stride_code - t_stride_x * 94273 - t_stride_y * 311)

    if transposed:
        stride_x = t_stride_x // l_stride_x
        stride_y = t_stride_y // l_stride_y
        stride_z = t_stride_z // l_stride_z
    else:
        stride_x = t_stride_x * l_stride_x
        stride_y = t_stride_y * l_stride_y
        stride_z = t_stride_z * l_stride_z
    
    return (94273 * stride_x + 311 * stride_y + stride_z)


def fapply(input: spTensor, fn: Callable[..., torch.Tensor], *args,
           **kwargs) -> spTensor:
    feats = fn(input.feats, *args, **kwargs)
    output = spTensor(coords=input.coords, feats=feats, buffer=input.buffer, \
        batchsize=input.batchsize, stride=input.stride, init_tag=input.init_tag, \
        coords_max=input.coords_max, coords_min=input.coords_min)
    output.cbook = input.cbook
    output.kmaps = input.kmaps
    return output


def cat(inputs: List[spTensor]) -> spTensor:
    feats = torch.cat([input.feats for input in inputs], dim=1)
    output = spTensor(coords=inputs[0].coords,
                          feats=feats,
                          buffer=inputs[0].buffer, 
                          batchsize=inputs[0].batchsize, 
                          stride=inputs[0].stride,
                          init_tag=inputs[0].init_tag, 
                          coords_max=inputs[0].coords_max, 
                          coords_min=inputs[0].coords_min)
    output.cbook = inputs[0].cbook
    output.kmaps = inputs[0].kmaps
    return output


def sparse_collate(inputs: List[spTensor]) -> spTensor:
    coords, feats = [], []
    stride = inputs[0].stride

    for k, x in enumerate(inputs):
        if isinstance(x.coords, np.ndarray):
            x.coords = torch.tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = torch.tensor(x.feats)

        assert isinstance(x.coords, torch.Tensor), type(x.coords)
        assert isinstance(x.feats, torch.Tensor), type(x.feats)
        assert x.stride == stride, (x.stride, stride)

        input_size = x.coords.shape[0]
        batch = torch.full((input_size, 1),
                           k,
                           device=x.coords.device,
                           dtype=torch.int)

        coords.append(torch.cat((batch, x.coords), dim=1))
        feats.append(x.feats)

    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    coords_min = [0, 0, 0]
    coords_max = ((torch.max(coords[:, 1:], dim=0).values).cpu().numpy()).tolist()
    output = spTensor(coords=coords, feats=feats, stride=stride, 
        batchsize=k+1, buffer=None, coords_min=coords_min, coords_max=coords_max)
    return output


def sparse_collate_fn(inputs: List[Any]) -> Any:
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_collate_fn(
                    [input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.cat(
                    [torch.tensor(input[name]) for input in inputs], dim=0)
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.cat([input[name] for input in inputs], dim=0)
            elif isinstance(inputs[0][name], spTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs


def make_ntuple(x: Union[int, List[int], Tuple[int, ...], torch.Tensor],
                ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, torch.Tensor):
        x = tuple(x.view(-1).cpu().numpy().tolist())

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x


def conv_info_encoder(info: List[int]) -> int:
    return 94273 * info[0] + 311 * info[1] + info[2]


def conv_info_decoder(code: int) -> Tuple[int, int, int]:
    code_x = code // 94273
    code_y = (code - code_x * 94273) // 311 
    code_z = (code - code_x * 94273 - code_y * 311)

    return (code_x, code_y, code_z)




                
            
        
    
