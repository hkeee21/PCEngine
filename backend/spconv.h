#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C"

void ConvolutionForward(const at::Tensor in_coords, const at::Tensor in_feats, 
                        const at::Tensor kernel, const int k_size, 
                        const at::Tensor in_map, at::Tensor out_feats,
                        const at::Tensor kernel_nnz, const at::Tensor whole_idx
                        );
