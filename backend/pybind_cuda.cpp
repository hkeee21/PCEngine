#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"
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


void conv_fwd_cuda(const at::Tensor in_feats, 
                const at::Tensor kernel, 
                const int k_size,
                at::Tensor out_feats,
                const at::Tensor kernel_nnz, 
                const at::Tensor in_map, 
                const at::Tensor out_map, 
                const at::Tensor in_csr, 
                const at::Tensor out_csr, 
                at::Tensor gather_buffer, 
                at::Tensor scatter_buffer, 
                const bool TensorCoreMode
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionForward(in_feats, 
                    kernel, 
                    k_size, 
                    out_feats, 
                    kernel_nnz, 
                    in_map, 
                    out_map, 
                    in_csr, 
                    out_csr, 
                    gather_buffer,
                    scatter_buffer,
                    TensorCoreMode);
}


void conv_bwd_cuda(const at::Tensor out_feats_grad, 
                    const at::Tensor in_feats, 
                    const at::Tensor kernel, 
                    const int k_size,
                    at::Tensor in_feats_grad, 
                    at::Tensor kernel_grad, 
                    const at::Tensor kernel_nnz, 
                    const at::Tensor kernel_pos,
                    const at::Tensor in_map, 
                    const at::Tensor out_map, 
                    at::Tensor in_buffer, 
                    at::Tensor out_buffer,   
                    const bool TensorCoreMode
                ){
  at::DeviceGuard guard(in_feats.device());
  
  ConvolutionBackward(out_feats_grad, 
                    in_feats, 
                    kernel, 
                    k_size,
                    in_feats_grad, 
                    kernel_grad, 
                    kernel_nnz, 
                    kernel_pos,
                    in_map, 
                    out_map, 
                    in_buffer, 
                    out_buffer,   
                    TensorCoreMode
                    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("mapping_cuda", &mapping_cuda);
  m.def("conv_fwd_cuda", &conv_fwd_cuda);
  m.def("conv_bwd_cuda", &conv_bwd_cuda);
}
