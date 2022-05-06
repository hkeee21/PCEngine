#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C"

void ConvolutionForward(at::Tensor in_coords, 
                        at::Tensor in_feats, 
                        at::Tensor kernel, 
                        const int k_size,
                        at::Tensor in_map,
                        at::Tensor out_feats);
