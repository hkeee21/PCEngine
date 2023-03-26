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
        batchsize=input.batchsize, stride=input.stride, init_tag=input.init_tag)
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
                          init_tag=inputs[0].init_tag)
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
    output = spTensor(coords=coords, feats=feats, stride=stride, 
        batchsize=k+1, buffer=None)
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


def vanillaConv(
                c_in: int, 
                c_out: int, 
                stride: list, 
                in_c, 
                in_f, 
                kv, 
                ks: list, 
                ):

    nnz = in_c.size(0)

    in_c = torch.as_tensor(in_c, dtype=int)
    stride = torch.as_tensor(stride, dtype=int)

    # print(stride)

    kofs_x = [(i - (ks[0] - 1) // 2) for i in range(ks[0])]
    kofs_y = [(i - (ks[1] - 1) // 2) for i in range(ks[1])]
    kofs_z = [(i - (ks[2] - 1) // 2) for i in range(ks[2])]

    if ((stride[0] == 1 or stride[0] == ks[0])
        and (stride[1] == 1 or stride[1] == ks[1])
        and (stride[2] == 1 or stride[2] == ks[2])): 

        print("easy downsampling.")
        
        out_c = torch.zeros((nnz, 4), dtype=int)

        out_c[:, 1:] = (torch.div(
            in_c[:, 1:],
            stride).trunc() * stride).int()
        out_c[:, 0] = in_c[:, 0]

        out_c = torch.unique(out_c, dim=0)

        out_nnz = out_c.size(0)

    else:
        
        print("expansion downsampling.")

        out_c = - torch.ones((nnz * kv.shape[0], 4),  dtype=int)

        out_nnz = 0
        for i in range(nnz):
            for x in range(ks[0]):
                if (in_c[i, 1] + kofs_x[x]) % stride[0] != 0 or \
                    (in_c[i, 1] + kofs_x[x]) < 0:
                    continue
                for y in range(ks[1]):
                    if (in_c[i, 2] + kofs_y[y]) % stride[1] != 0 or \
                        (in_c[i, 2] + kofs_y[y]) < 0:
                        continue
                    for z in range(ks[2]):
                        if (in_c[i, 3] + kofs_y[z]) % stride[2] != 0 or \
                            (in_c[i, 3] + kofs_y[z]) < 0:
                            continue
                        out_c[out_nnz, 0] = in_c[i, 0]
                        out_c[out_nnz, 1] = in_c[i, 1] + kofs_x[x]
                        out_c[out_nnz, 2] = in_c[i, 2] + kofs_y[y]
                        out_c[out_nnz, 3] = in_c[i, 3] + kofs_y[z]
                        out_nnz += 1

        out_c = out_c[0:out_nnz, :]

        out_c = torch.unique(out_c, dim=0)

        out_nnz = out_c.size(0)

    print("vanilla out nnz:", out_nnz)
    # out_c[:, 1:] = (torch.div(
    #         in_c[:, 1:],
    #         stride).trunc() * stride).int()
    # out_c[:, 0] = in_c[:, 0]
            
    # out_c = torch.unique(out_c, dim=0)

    # out_nnz = out_c.size(0)

    
    '''for i in range(out_nnz):
        print("(%d, %d, %d, %d)" % (out_c[i, 0], out_c[i, 1], out_c[i, 2], out_c[i, 3]))'''

    out = np.zeros((out_nnz, c_out))

    print("%d-%d" % (nnz, out_nnz))

    '''
    for i in range(nnz):
        print("(%d, %d, %d) " % (in_c[i, 0], in_c[i, 1], in_c[i, 2]))
    print("------------")
    for j in range(out_nnz):
        print("(%d, %d, %d) " % (out_c[j, 0], out_c[j, 1], out_c[j, 2]))'''

    sum_nnz = 0

    # input loop
    for i in range(nnz):
        # output loop
        for j in range(out_nnz):
            if in_c[i, 0] == out_c[j, 0]:
                off_x = in_c[i, 1] - out_c[j, 1]
                off_y = in_c[i, 2] - out_c[j, 2]
                off_z = in_c[i, 3] - out_c[j, 3]
                if (off_x in kofs_x and off_y in kofs_y and off_z in kofs_z):
                    sum_nnz += 1
                    kid = (off_x + (ks[0] - 1) // 2) * ks[1] * ks[2] \
                        + (off_y + (ks[1] - 1) // 2) * ks[2] \
                        + (off_z + (ks[2] - 1) // 2)
                    # Conv operation
                    for co in range(c_out):
                        tv = 0
                        for c in range(c_in):
                            tv += kv[kid, c, co] * in_f[i, c]
                    
                        out[j, co] += tv
    
    print("vanilla sum nnz: %d" % sum_nnz)
    
    return out


def vanillaConvBackward(innz: int,
                        onnz: int, 
                        c_in: int, 
                        c_out: int, 
                        in_c, 
                        out_c, 
                        in_f,
                        kw, 
                        ks:list,
                        out_f_g):
    
    in_f_g = np.zeros_like(in_f)
    kw_g = np.zeros_like(kw)

    print(in_c.shape)
    print(out_c.shape)
    print(in_f.shape)
    print(kw.shape)
    print(out_f_g.shape)

    kofs_x = [(i - (ks[0] - 1) // 2) for i in range(ks[0])]
    kofs_y = [(i - (ks[1] - 1) // 2) for i in range(ks[1])]
    kofs_z = [(i - (ks[2] - 1) // 2) for i in range(ks[2])]

    for i in range(innz):
        for j in range(onnz):
            if in_c[i, 0] == out_c[j, 0]:
                off_x = in_c[i, 1] - out_c[j, 1]
                off_y = in_c[i, 2] - out_c[j, 2]
                off_z = in_c[i, 3] - out_c[j, 3]
                if (off_x in kofs_x and off_y in kofs_y and off_z in kofs_z):

                    kid = (off_x + (ks[0] - 1) // 2) * ks[1] * ks[2] \
                        + (off_y + (ks[1] - 1) // 2) * ks[2] \
                        + (off_z + (ks[2] - 1) // 2)
                    
                    # W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
                    for c in range(c_in):
                        tv = 0
                        for co in range(c_out):
                            tv += kw[kid, c, co] * out_f_g[j, co]
                        in_f_g[i, c] += tv
                
                    # {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
                    for c in range(c_in):
                        for co in range(c_out):
                            kw_g[kid, c, co] += in_f[i, c] * out_f_g[j, co]
    
    return in_f_g, kw_g


def CheckResults(len: int, 
                c_out: int, 
                results1, 
                results2):
    accum_error = 0
    for i in range(len):
        n = i // c_out
        c = i % c_out
        error = abs(results1[n, c] - results2[n, c])
        if (error > 0.05):
            print("The %d-th nnz's %d-th channel has abs error: %.4f (%.4f, %.4f)" % (n, c, error, results1[n, c], results2[n, c]))
        
        accum_error += error
    

    return accum_error


def CheckResultsWeight(k_vol: int, 
                    c_in: int, 
                    c_out: int, 
                    results1, 
                    results2):
    
    accum_error = 0
    for i in range(k_vol):
        for c in range(c_in):
            for co in range(c_out):
                error = abs(results1[i, c, co] - results2[i, c, co])
                if (error > 0.01):
                    print("The %d-th weight's %d-th input channel %d-th output channel has abs error: %.4f (%.4f, %.4f)" \
                        % (i, c, co, error, results1[i, c, co], results2[i, c, co]))
                
                accum_error += error
    
    return accum_error


def binary_search(
    csr, eid: int, start: int, end: int
):

    lo = start
    hi = end
    if (lo == hi):
        return lo
    while (lo < hi) :
        mid = floor((lo + hi) / 2)
        if (csr[mid] <= eid) :
            lo = mid + 1
        else:
            hi = mid
    
    if (csr[hi] <= eid):
        target = hi
    else:
        target = hi - 1

    if (csr[target + 1] - eid < eid - csr[target]):
        target = target + 1
    
    return target



                
            
        
    
