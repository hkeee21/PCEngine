#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C"

at::Tensor HashMapD1(
                const at::Tensor in_coords, 
                const int k_size_code, 
                const int k_vol, 
                const int c_in, 
                const int c_out, 
                const int l_stride_code, 
                const int t_stride_code, 
                const int padding_code, 
                const int min_x,
                const int min_y,
                const int min_z,
                const int max_x,
                const int max_y,
                const int max_z,
                at::Tensor imap,
                at::Tensor omap,  
                at::Tensor icsr,
                at::Tensor ocsr,
                at::Tensor kernel_nnz,
                at::Tensor kernel_kpos,
                at::Tensor kernel_qkpos, 
                const bool separate_mid
                );


at::Tensor HashMapD2(
                const at::Tensor in_coords, 
                const int batch_size, 
                const int k_size_code, 
                const int k_vol, 
                const int c_in, 
                const int c_out, 
                const int l_stride_code, 
                const int t_stride_code, 
                const int padding_code, 
                const int min_x,
                const int min_y,
                const int min_z,
                const int max_x,
                const int max_y,
                const int max_z,
                at::Tensor map,
                at::Tensor kernel_nnz,
                at::Tensor kernel_kpos,
                at::Tensor kernel_qkpos, 
                const bool separate_mid
                );