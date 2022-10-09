#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C"

at::Tensor HashMap(
                const at::Tensor in_coords, 
                const int k_size_code, 
                const int k_vol, 
                const int c_in, 
                const int c_out, 
                const int l_stride_code, 
                const int t_stride_code, 
                at::Tensor imap,
                at::Tensor omap,  
                at::Tensor icsr,
                at::Tensor ocsr,
                at::Tensor kernel_nnz,
                at::Tensor kernel_pos,
                const bool separate_mid
                );
