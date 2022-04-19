#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convolution_forward_cuda", &convolution_forward_cuda);
  m.def("convolution_backward_cuda", &convolution_backward_cuda);
}
