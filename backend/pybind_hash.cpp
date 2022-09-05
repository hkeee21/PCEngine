#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "hash.h"

int mapping_cuda(at::Tensor in_coords, 
            const int k_size,
            at::Tensor imap,
            at::Tensor omap, 
            at::Tensor icsr,
            at::Tensor ocsr, 
            at::Tensor kernel_nnz,
            at::Tensor kernel_pos
            ){
  at::DeviceGuard guard(in_coords.device());

  return HashMap(in_coords, k_size, imap, omap, icsr, ocsr, kernel_nnz, kernel_pos);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &mapping_cuda);
}
