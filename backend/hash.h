#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C"

at::Tensor HashMap(at::Tensor in_coords, 
                const int k_size,
                at::Tensor in_map,
                at::Tensor kernel_nnz
                );
