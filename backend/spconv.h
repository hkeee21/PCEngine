#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C"

void ConvolutionForward(const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int kernel_size_code, 
                        const int sum_nnz, 
                        at::Tensor out_feats, 
                        const at::Tensor kernel_nnz,
                        const at::Tensor kernel_pos, 
                        const at::Tensor in_map, 
                        const at::Tensor out_map,
                        const at::Tensor in_csr, 
                        const at::Tensor out_csr, 
                        at::Tensor buffer, 
                        const bool separate_mid, 
                        const bool TensorCoreMode
                        );


void ConvolutionBackward(const at::Tensor out_feats_grad, 
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int k_size_code,
                        const int sum_nnz, 
                        at::Tensor in_feats_grad, 
                        at::Tensor kernel_grad, 
                        const at::Tensor kernel_nnz, 
                        const at::Tensor kernel_pos,
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const at::Tensor in_csr, 
                        const at::Tensor out_csr, 
                        at::Tensor buffer, 
                        const bool TensorCoreMode
                        );
