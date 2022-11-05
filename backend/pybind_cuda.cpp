#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"
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


void conv_fwd_cuda(const at::Tensor in_feats, 
                const at::Tensor kernel, 
                const int kernel_size_code, 
                const int sum_nnz, 
                at::Tensor out_feats,
                const at::Tensor kernel_nnz, 
                const at::Tensor kernel_pos, 
                const at::Tensor in_map, 
                const at::Tensor out_map, 
                const at::Tensor in_csr, 
                const at::Tensor out_csr, 
                at::Tensor buffer, 
                const bool separate_mid, 
                const bool TensorCoreMode
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionForward(in_feats, kernel, kernel_size_code, sum_nnz, 
    out_feats, kernel_nnz, kernel_pos, in_map, out_map, 
    in_csr, out_csr, buffer, separate_mid, TensorCoreMode);
}


void conv_bwd_cuda(const at::Tensor out_feats_grad, 
                    const at::Tensor in_feats, 
                    const at::Tensor kernel, 
                    const int k_size_code,
                    const int sum_nnz, 
                    at::Tensor in_feats_grad, 
                    at::Tensor kernel_grad, 
                    const at::Tensor kernel_nnz, 
                    const at::Tensor kernel_pos,
                    const at::Tensor in_map, 
                    const at::Tensor out_map, 
                    const at::Tensor in_csr, 
                    const at::Tensor out_csr,
                    at::Tensor buffer, 
                    const bool TensorCoreMode
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionBackward(out_feats_grad, in_feats, 
      kernel, k_size_code, sum_nnz, in_feats_grad, 
      kernel_grad, kernel_nnz, kernel_pos, in_map, 
      out_map, in_csr, out_csr, buffer, TensorCoreMode
      );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &mapping_cuda);
  m.def("conv_fwd_cuda", &conv_fwd_cuda);
  m.def("conv_bwd_cuda", &conv_bwd_cuda);
}
