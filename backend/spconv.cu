#include "spconv.h"
#include "gemm.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using namespace std;

#define GEMM_SIZE 16
#define PAR_THREAD 256
#define LoopGemm 0
#define FuseGemm 1
#define FuseOrch 2

extern "C"

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)


void ConvolutionForward(const at::Tensor in_coords, const at::Tensor in_feats, 
                        const at::Tensor kernel, const int k_size, 
                        const at::Tensor in_map, at::Tensor out_feats,
                        const at::Tensor kernel_nnz, const at::Tensor whole_idx
                        ){
    
    // printf("[SubmanifoldSparseConv] - Starts.\n");

    int nnz = in_coords.size(0);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    int k_vol = k_size * k_size * k_size;

    float *in_feats_ptr = in_feats.data_ptr<float>();
    float *weight_ptr = kernel.data_ptr<float>();
    float *out_feats_ptr = out_feats.data_ptr<float>();
    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *in_map_ptr = in_map.data_ptr<int>();
    long *whole_idx_ptr = whole_idx.data_ptr<long>();

    int total_nnz = kernel_nnz.sum().item<int>();

    unsigned int branch = LoopGemm;

    // branch
    if (total_nnz < 45000){
        if (in_channel <= 64 && out_channel <= 128){
            branch = FuseGemm;
        }
        else{ branch = FuseOrch;}
    }

    
    if (branch == LoopGemm){

        int max_nnz = kernel_nnz.max().item<int>();

        at::Tensor gfeats = torch::zeros({max_nnz * in_channel}, at::device(in_feats.device()).dtype(at::ScalarType::Float));
        at::Tensor sfeats = torch::zeros({max_nnz * out_channel}, at::device(in_feats.device()).dtype(at::ScalarType::Float));

        float *gfeats_ptr = gfeats.data_ptr<float>();
        float *sfeats_ptr = sfeats.data_ptr<float>();

        // cublas
        const float alpha = 1.0;
        const float beta = 0.0;

        cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        CUBLAS_CHECK(cublasSetStream(cublasH, stream));

        // loop over all kernel offsets
        int cur_idx = 0;

        // Suppose an odd kernel size
        for (int i = 0; i < k_vol; i++){

            int cur_nnz = kernel_nnz[i].item<int>();

            if (cur_nnz == 0){continue;}

            size_t const block_g = in_channel > PAR_THREAD ? in_channel : PAR_THREAD;
            size_t const grid_g = (cur_nnz * in_channel + block_g - 1) / block_g;

            gather<<<grid_g, block_g, ((block_g + in_channel - 1) / in_channel + 1) * sizeof(int)>>>(
                nnz,
                cur_nnz,
                in_channel,
                in_feats_ptr,
                &whole_idx_ptr[cur_idx],
                &in_map_ptr[i * nnz],
                gfeats_ptr
            );

            // cublas Sgemm for matmul
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                out_channel, cur_nnz, in_channel, 
                &alpha, 
                &weight_ptr[i * in_channel * out_channel], 
                out_channel, 
                gfeats_ptr, 
                in_channel, 
                &beta, 
                sfeats_ptr, 
                out_channel)
            );

            size_t const block_s = out_channel > PAR_THREAD ? out_channel : PAR_THREAD;
            size_t const grid_s = (cur_nnz * out_channel + block_s - 1) / block_s;
        
            scatter<<<grid_s, block_s, ((block_s + out_channel - 1) / out_channel + 1) * sizeof(int)>>>(
                nnz,
                cur_nnz,
                out_channel,
                sfeats_ptr, 
                &whole_idx_ptr[cur_idx],
                &in_map_ptr[i * nnz],
                out_feats_ptr
            );

            cur_idx += cur_nnz;
        }

    }
    else if (branch == FuseOrch){

        at::Tensor gfeats = torch::zeros({total_nnz * in_channel}, at::device(in_feats.device()).dtype(at::ScalarType::Float));
        at::Tensor sfeats = torch::zeros({total_nnz * out_channel}, at::device(in_feats.device()).dtype(at::ScalarType::Float));

        float *gfeats_ptr = gfeats.data_ptr<float>();
        float *sfeats_ptr = sfeats.data_ptr<float>();

        // cublas
        const float alpha = 1.0;
        const float beta = 0.0;

        cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        CUBLAS_CHECK(cublasSetStream(cublasH, stream));

        size_t const block_g = in_channel > PAR_THREAD ? in_channel : PAR_THREAD;
        size_t const grid_g = (total_nnz * in_channel + block_g - 1) / block_g;

        // gather for all kernel offsets
        gather_all<<<grid_g, block_g, ((block_g + in_channel - 1) / in_channel + 1) * sizeof(int)>>>(
            nnz,
            total_nnz,
            in_channel,
            in_feats_ptr,
            whole_idx_ptr,
            in_map_ptr,
            gfeats_ptr
        );

        // loop over all kernel offsets
        int cur_idx = 0;

        // Suppose an odd kernel size
        for (int i = 0; i < k_vol; i++){

            int cur_nnz = kernel_nnz[i].item<int>();

            if (cur_nnz == 0){continue;}

            // cublas Sgemm for matmul
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                out_channel, cur_nnz, in_channel, 
                &alpha, 
                &weight_ptr[i * in_channel * out_channel], 
                out_channel, 
                &gfeats_ptr[cur_idx * in_channel], 
                in_channel, 
                &beta, 
                &sfeats_ptr[cur_idx * out_channel], 
                out_channel)
            );
                
            cur_idx += cur_nnz;
        }

        size_t const block_s = out_channel > PAR_THREAD ? out_channel : PAR_THREAD;
        size_t const grid_s = (total_nnz * out_channel + block_s - 1) / block_s;

        // scatter for all kernel offsets
        scatter_all<<<grid_s, block_s, ((block_s + out_channel - 1) / out_channel + 1) * sizeof(int)>>>(
            nnz,
            total_nnz,
            out_channel,
            sfeats_ptr, 
            whole_idx_ptr,
            in_map_ptr,
            out_feats_ptr
        );

    }
    else if (branch == FuseGemm){

        // loop over all kernel offsets
        int cur_idx = 0;

        // Suppose an odd kernel size
        for (int i = 0; i < k_vol; i++){

            int cur_nnz = kernel_nnz[i].item<int>();

            if (cur_nnz == 0){continue;}

            size_t const gridnum_x = (out_channel + GEMM_SIZE - 1) / GEMM_SIZE;
            size_t const gridnum_y = (cur_nnz + GEMM_SIZE - 1) / GEMM_SIZE;

            // GEMM
            gemm<<<dim3(gridnum_x, gridnum_y, 1), dim3(GEMM_SIZE, GEMM_SIZE, 1)>>>(
                    nnz, 
                    cur_nnz, 
                    in_channel, out_channel,
                    in_feats_ptr,
                    &weight_ptr[i * in_channel * out_channel],
                    out_feats_ptr,
                    &whole_idx_ptr[cur_idx], 
                    &in_map_ptr[i * nnz]);
        
            cur_idx += cur_nnz;
        }

    }
    else{ printf("No Mode Matches !");}

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return;
}
