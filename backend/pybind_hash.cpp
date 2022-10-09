#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "hash.h"

at::Tensor mapping_cuda(
                const at::Tensor in_coords, 
                const int k_size_code, 
                const int k_vol, 
                const int c_in, 
                const int c_out, 
                const int l_stride_code, 
                const int t_stride_code, 
                at::Tensor imap,
                at::Tensor omap,  
                at::Tensor icsr,
                at::Tensor ocsr,
                at::Tensor kernel_nnz,
                at::Tensor kernel_pos,
                const bool sep_mid_computation
                ){
  at::DeviceGuard guard(in_coords.device());

  return HashMap(in_coords, k_size_code, k_vol, c_in, c_out, 
    l_stride_code, t_stride_code, imap, omap, icsr, ocsr, 
    kernel_nnz, kernel_pos, sep_mid_computation);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &mapping_cuda);
}
