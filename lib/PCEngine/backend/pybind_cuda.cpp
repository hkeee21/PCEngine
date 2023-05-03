#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"
#include "hash.h" 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &HashMap);
  m.def("mapping_simple_cuda", &HashMap_simple);
  m.def("conv_fwd_cuda", &ConvolutionForward);
  m.def("conv_fwd_simple_cuda", &ConvolutionForward_simple);
  m.def("conv_fwd_naive_cuda", &ConvolutionForward_naive);
  m.def("conv_fwd_batched_cuda", &ConvolutionForward_batched);
  m.def("conv_fwd_separate_cuda", &ConvolutionForward_separate);
  m.def("conv_bwd_cuda", &ConvolutionBackward);
  m.def("gather_coded_CSR_cuda", &gather_with_coded_CSR_wrapper);
  m.def("scatter_coded_CSR_cuda", &scatter_with_coded_CSR_wrapper);
  m.def("gather_vanilla_cuda", &gather_without_coded_CSR_wrapper);
  m.def("scatter_vanilla_cuda", &scatter_without_coded_CSR_wrapper);
  m.def("map_to_matrix_cuda", &map_to_matrix_wrapper);
  m.def("torchsparse_gather_cuda", &torchsparse_gather_wrapper);
  m.def("torchsparse_scatter_cuda", &torchsparse_scatter_wrapper);
  m.def("mapping_coded_csr_cuda", &HashMap_coded_CSR);
}
