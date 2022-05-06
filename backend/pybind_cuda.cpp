#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"

void conv_fwd_cuda(at::Tensor in_coords, 
                        at::Tensor in_feats, 
                        at::Tensor kernel, 
                        const int k_size,
                        at::Tensor in_map,
                        at::Tensor out_feats){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionForward(in_coords, in_feats, kernel, k_size, in_map, out_feats);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_fwd_cuda", &conv_fwd_cuda);
}
