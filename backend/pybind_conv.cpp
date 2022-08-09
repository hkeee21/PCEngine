#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"

void conv_fwd_cuda(const at::Tensor in_feats, 
                const at::Tensor kernel, 
                const int k_size,
                at::Tensor out_feats,
                const at::Tensor kernel_nnz, 
                const at::Tensor kernel_pos, 
                const at::Tensor in_map, 
                const at::Tensor out_map, 
                at::Tensor gather_buffer, 
                at::Tensor scatter_buffer, 
                const bool TensorCoreMode
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionForward(in_feats, 
                    kernel, 
                    k_size, 
                    out_feats, 
                    kernel_nnz, 
                    kernel_pos, 
                    in_map, 
                    out_map, 
                    gather_buffer,
                    scatter_buffer,
                    TensorCoreMode);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("conv_fwd_cuda", &conv_fwd_cuda);
}
