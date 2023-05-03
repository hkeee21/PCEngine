import open3d as o3d
from typing import List, Tuple, Union, Any
from itertools import repeat
import numpy as np
import torch
import spconv.pytorch as spconv


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


def sparse_collate(inputs: List[spconv.SparseConvTensor]) -> spconv.SparseConvTensor:
    coords, feats = [], []
    temp_shape = []

    for k, x in enumerate(inputs):
        if isinstance(x.indices, np.ndarray):
            x.indices = torch.tensor(x.indices)
        if isinstance(x.features, np.ndarray):
            x.features = torch.tensor(x.features)

        assert isinstance(x.indices, torch.Tensor), type(x.indices)
        assert isinstance(x.features, torch.Tensor), type(x.features)

        x.indices = x.indices[:, 1:]

        input_size = x.indices.shape[0]
        batch = torch.full((input_size, 1),
                           k,
                           device=x.indices.device,
                           dtype=torch.int)

        coords.append(torch.cat((batch, x.indices), dim=1))
        feats.append(x.features)
        temp_shape.append(torch.as_tensor(x.spatial_shape).reshape(1, 3))

    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    temp_shape = torch.cat(temp_shape, dim=0)
    # shape = np.max(temp_shape.numpy(), axis=0)
    shape = (torch.max(coords[:, 1:], dim=0).values + 10).cpu().numpy()
    for i in range(3):
        if shape[i] < 31:
            shape[i] = 31

    output = spconv.SparseConvTensor(indices=coords, features=feats, batch_size=k+1, spatial_shape=shape)
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
            elif isinstance(inputs[0][name], spconv.SparseConvTensor):
                output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs


def vanillaConv(nnz: int, 
                c_in: int, 
                c_out: int, 
                in_c, 
                in_f, 
                kv, 
                ks: int, 
                ):
    
    out = np.zeros((nnz, c_out))

    for i in range(nnz):
        for j in range(nnz):
            off_x = in_c[i, 0] - in_c[j, 0]
            off_y = in_c[i, 1] - in_c[j, 1]
            off_z = in_c[i, 2] - in_c[j, 2]
            if (abs(off_x) <= ks // 2 and abs(off_y) <= ks // 2 and abs(off_z) <= ks // 2):
                kid = (off_x + ks // 2) * ks * ks + (off_y + ks // 2) * ks + off_z + ks // 2
                # Conv operation
                for co in range(c_out):
                    tv = 0
                    for c in range(c_in):
                        tv += kv[kid, c, co] * in_f[i, c]
                    
                    out[j, co] += tv
    
    return out


def CheckResults(len: int, 
                c_out: int, 
                results1, 
                results2):
    accum_error = 0
    for i in range(len):
        n = i // c_out
        c = i % c_out
        error = abs(results1[n, c] - results2[n, c])
        if (error > 0.01):
            print("The %d-th nnz's %d-th channel has abs error: %.4f" % (n, c, error))
        
        accum_error += error
    

    return accum_error

                
            
        
    
