#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "mmio.hpp"
#include "util.hpp"
#include "spconv.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_fwd_cuda", &conv_fwd_cuda);
}
