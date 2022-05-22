#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"
#include "hash.h"

at::Tensor mapping_cuda(at::Tensor in_coords, 
            const int k_size,
            at::Tensor in_map,
            at::Tensor kernel_nnz
            ){
  at::DeviceGuard guard(in_coords.device());

  return HashMap(in_coords, k_size, in_map, kernel_nnz);
}


void conv_fwd_cuda(at::Tensor in_coords, 
                at::Tensor in_feats, 
                at::Tensor kernel, 
                const int k_size,
                at::Tensor in_map, 
                at::Tensor out_feats,
                at::Tensor kernel_nnz, 
                at::Tensor whole_idx
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionForward(in_coords, in_feats, kernel, k_size, in_map, out_feats, kernel_nnz, whole_idx);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &mapping_cuda);
  m.def("conv_fwd_cuda", &conv_fwd_cuda);
}
