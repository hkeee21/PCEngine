#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"

void conv_fwd_cuda(const at::Tensor in_feats, 
                const at::Tensor kernel, 
                const int k_size,
                const int sum_nnz, 
                at::Tensor out_feats,
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
  
  ConvolutionForward(in_feats, kernel, k_size, sum_nnz, 
    out_feats, kernel_nnz, kernel_pos, in_map, out_map, 
    in_csr, out_csr, buffer, TensorCoreMode);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("conv_fwd_cuda", &conv_fwd_cuda);
}
