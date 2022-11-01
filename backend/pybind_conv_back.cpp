#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"

void conv_bwd_cuda(const at::Tensor out_feats_grad, 
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
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionBackward(out_feats_grad, in_feats, 
      kernel, k_size_code, sum_nnz, in_feats_grad, 
      kernel_grad, kernel_nnz, kernel_pos, in_map, 
      out_map, in_csr, out_csr, buffer, TensorCoreMode
      );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("conv_bwd_cuda", &conv_bwd_cuda);
}