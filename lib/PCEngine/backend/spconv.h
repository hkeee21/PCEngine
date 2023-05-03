#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C"

void ConvolutionForward(at::Tensor in_feats, 
                        at::Tensor kernel, 
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


void ConvolutionForward_simple(
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int ksize_code, 
                        const int sum_nnz, 
                        at::Tensor out_feats, 
                        const at::Tensor kpos, 
                        const at::Tensor qkpos, 
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const bool separate_mid, 
                        const bool TensorCoreFast
                        );


void ConvolutionForward_naive(
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int ksize_code, 
                        const int sum_nnz, 
                        at::Tensor out_feats, 
                        const at::Tensor kernel_nnz, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const bool separate_mid, 
                        const bool TensorCoreFast
                        );


void ConvolutionForward_batched(
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int ksize_code, 
                        const int sum_nnz, 
                        at::Tensor out_feats, 
                        const at::Tensor kernel_nnz, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const bool separate_mid, 
                        const int M,
                        const float theta
                        );


void ConvolutionForward_separate(
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int ksize_code, 
                        const int sum_nnz, 
                        at::Tensor out_feats, 
                        const at::Tensor kernel_nnz, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const bool separate_mid
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


void gather_with_coded_CSR_wrapper(
                        at::Tensor in_feats, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor in_map, 
                        const at::Tensor in_csr, 
                        at::Tensor buffer);


void scatter_with_coded_CSR_wrapper( 
                        const int buffer_offset, 
                        at::Tensor out_feats, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor out_map, 
                        const at::Tensor out_csr, 
                        at::Tensor buffer);


void gather_without_coded_CSR_wrapper(
                        const int k_vol, 
                        at::Tensor in_feats, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor in_map, 
                        at::Tensor buffer);


void scatter_without_coded_CSR_wrapper( 
                        const int buffer_offset, 
                        const int k_vol, 
                        at::Tensor out_feats, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor out_map, 
                        at::Tensor buffer);


void map_to_matrix_wrapper(
                        const int nnz, 
                        const int k_vol, 
                        at::Tensor csr, 
                        at::Tensor map, 
                        at::Tensor matrix);


void torchsparse_gather_wrapper(
                        at::Tensor in_feat, 
                        at::Tensor buffer, 
                        const int kernel_volume, 
                        at::Tensor kpos, 
                        at::Tensor input_mask, 
                        at::Tensor output_mask, 
                        bool transpose, 
                        bool precompute_mid
                        );


void torchsparse_scatter_wrapper(
                        at::Tensor out_feat, 
                        at::Tensor buffer, 
                        const int buffer_offset, 
                        const int kernel_volume, 
                        at::Tensor kpos, 
                        at::Tensor input_mask, 
                        at::Tensor output_mask, 
                        bool transpose, 
                        bool precompute_mid
                        );