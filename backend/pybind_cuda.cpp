#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"
#include "hash.h" 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_d1_cuda", &HashMapD1);
  m.def("mapping_d2_cuda", &HashMapD2);
  m.def("conv_fwd_d1_cuda", &ConvolutionForwardD1);
  m.def("conv_fwd_d2_cuda", &ConvolutionForwardD2);
  m.def("conv_bwd_d1_cuda", &ConvolutionBackwardD1);
}
