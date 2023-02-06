#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"
#include "hash.h" 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &HashMap);
  m.def("mapping_simple_cuda", &HashMap_simple);
  m.def("conv_fwd_cuda", &ConvolutionForward);
  m.def("conv_fwd_simple_cuda", &ConvolutionForwardBlockFused);
  m.def("conv_bwd_cuda", &ConvolutionBackward);
}
