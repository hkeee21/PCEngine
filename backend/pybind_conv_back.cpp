#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"

void conv_bwd_cuda(const at::Tensor out_feats_grad, 
                    const at::Tensor in_feats, 
                    const at::Tensor kernel, 
                    const int k_size,
                    at::Tensor in_feats_grad, 
                    at::Tensor kernel_grad, 
                    const at::Tensor kernel_nnz, 
                    const at::Tensor kernel_pos,
                    const at::Tensor in_map, 
                    const at::Tensor out_map, 
                    at::Tensor in_buffer, 
                    at::Tensor out_buffer,   
                    const bool TensorCoreMode
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionBackward(out_feats_grad, 
                    in_feats, 
                    kernel, 
                    k_size,
                    in_feats_grad, 
                    kernel_grad, 
                    kernel_nnz, 
                    kernel_pos,
                    in_map, 
                    out_map, 
                    in_buffer, 
                    out_buffer,   
                    TensorCoreMode
                    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("conv_bwd_cuda", &conv_bwd_cuda);
}