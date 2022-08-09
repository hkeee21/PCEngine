#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "hash.h"

void mapping_cuda(at::Tensor in_coords, 
            const int k_size,
            at::Tensor imap,
            at::Tensor omap, 
            at::Tensor kernel_nnz
            ){
  at::DeviceGuard guard(in_coords.device());

  HashMap(in_coords, k_size, imap, omap, kernel_nnz);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &mapping_cuda);
}
