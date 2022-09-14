#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"

void conv_fwd_cuda_hashmap(
                const at::Tensor in_coords, 
                const at::Tensor in_feats, 
                const at::Tensor kernel, 
                const int k_size, 
                at::Tensor imap,
                at::Tensor omap,  
                at::Tensor icsr,
                at::Tensor ocsr,
                at::Tensor kernel_nnz, 
                at::Tensor kernel_pos, 
                at::Tensor in_buffer, 
                at::Tensor out_buffer, 
                at::Tensor out_feats, 
                const bool TensorCoreMode
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionForwardwithHashmap(
    in_coords, 
    in_feats, 
    kernel, 
    k_size, 
    imap,
    omap,  
    icsr,
    ocsr,
    kernel_nnz, 
    kernel_pos, 
    in_buffer, 
    out_buffer, 
    out_feats, 
    TensorCoreMode);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("conv_fwd_cuda_hashmap", &conv_fwd_cuda_hashmap);
}
