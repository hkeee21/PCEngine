import open3d as o3d
from typing import List, Tuple, Union
from itertools import repeat
import numpy as np
import torch

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


def vanillaConvBackward(nnz: int,
                        c_in: int, 
                        c_out: int, 
                        in_c, 
                        in_f,
                        kw, 
                        ks:int,
                        out_f_g):
    
    in_f_g = np.zeros_like(in_f)
    kw_g = np.zeros_like(kw)

    for i in range(nnz):
        for j in range(nnz):
            off_x = in_c[i, 0] - in_c[j, 0]
            off_y = in_c[i, 1] - in_c[j, 1]
            off_z = in_c[i, 2] - in_c[j, 2]
            if (abs(off_x) <= ks // 2 and abs(off_y) <= ks // 2 and abs(off_z) <= ks // 2):
                kid = (off_x + ks // 2) * ks * ks + (off_y + ks // 2) * ks + off_z + ks // 2
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
        if (error > 0.01):
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
        



                
            
        
    
