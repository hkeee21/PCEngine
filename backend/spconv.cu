#include "spconv.h"
#include "spconv.cuh"

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

#define PAR_THREAD 256
#define DIV_UP(x, y) (x + y - 1) / y
#define SHM_CAL(n, k) n * (k - 1) + 1   


extern "C"

void ConvolutionForward(const at::Tensor in_feats, 
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
    
    // printf("[SubmanifoldSparseConv] - Starts.\n");

    int nnz = in_feats.size(0);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    const int k_vol = k_size * k_size * k_size;

    float *in_feats_ptr = in_feats.data_ptr<float>();
    float *weight_ptr = kernel.data_ptr<float>();
    float *out_feats_ptr = out_feats.data_ptr<float>();
    int *in_map_ptr = in_map.data_ptr<int>();
    int *out_map_ptr = out_map.data_ptr<int>();
    int *in_csr_ptr = in_csr.data_ptr<int>();
    int *out_csr_ptr = out_csr.data_ptr<int>();

    // int *kernel_pos_ptr = kernel_pos.data_ptr<int>();

    // int max_nnz = kernel_nnz.max().item<int>();
    // int max_nnz_mod = ceil((double)max_nnz / (double)WMMA_SIZE) * WMMA_SIZE;
    // int sum_nnz = gather_buffer.size(0);

    float *gfeats_ptr = gather_buffer.data_ptr<float>();
    float *sfeats_ptr = scatter_buffer.data_ptr<float>();

    // cublas
    const float alpha = 1.0;
    const float beta = 0.0;
 
    cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // stride = 1
    int mid_nnz = nnz;

    // computation for w[0, 0, 0]
    if (TensorCoreMode){

        CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
        
        CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, mid_nnz, in_channel, 
                    &alpha, 
                    &weight_ptr[k_vol / 2 * in_channel * out_channel], CUDA_R_32F, out_channel, 
                    in_feats_ptr, CUDA_R_32F, in_channel, 
                    &beta, 
                    out_feats_ptr, CUDA_R_32F, out_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else{
        CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, mid_nnz, in_channel, 
                    &alpha, 
                    &weight_ptr[k_vol / 2 * in_channel * out_channel], 
                    out_channel, 
                    in_feats_ptr, 
                    in_channel, 
                    &beta, 
                    out_feats_ptr, 
                    out_channel));
    }

    // gather features
    // size_t const BLOCK_NUM = DIV_UP(nnz, 4);

    /*gather_all_input_major_template2<4, 2, 32>
                <<<BLOCK_NUM, dim3(32, 2, 4), 
                (4 * (k_vol - 1) + 1) * sizeof(int)>>>(
                nnz, k_vol, sum_nnz, kernel_pos_ptr, in_channel,
                in_feats_ptr, in_map_ptr, gfeats_ptr);*/

    /*switch(k_size){
        case 5:
            gather_all_input_major_template2<4, 2, 32, 4 * 124 + 1>
                <<<BLOCK_NUM, dim3(32, 2, 4)>>>(
                nnz, k_vol, sum_nnz, kernel_pos_ptr, in_channel,
                in_feats_ptr, in_map_ptr, gfeats_ptr);
            break;
        case 3:
            gather_all_input_major_template2<4, 2, 32, 4 * 26 + 1>
                <<<BLOCK_NUM, dim3(32, 2, 4)>>>(
                nnz, k_vol, sum_nnz, kernel_pos_ptr, in_channel,
                in_feats_ptr, in_map_ptr, gfeats_ptr);
            break;
        case 2:
            gather_all_input_major_template2<4, 2, 32, 4 * 7 + 1>
                <<<BLOCK_NUM, dim3(32, 2, 4)>>>(
                nnz, k_vol, sum_nnz, kernel_pos_ptr, in_channel,
                in_feats_ptr, in_map_ptr, gfeats_ptr);
            break;
    }*/

    // size_t const block_g = in_channel > PAR_THREAD ? in_channel : PAR_THREAD;
    // size_t const grid_g = ((nnz) * (in_channel) + block_g - 1) / block_g;

    /*gather_all_input_major_template<4, 2, 32>
        <<<DIV_UP(nnz, 4), dim3(32, 2, 4)>>>(
            nnz,
            k_vol, 
            sum_nnz,
            kernel_pos_ptr, 
            in_channel,
            in_feats_ptr,
            in_map_ptr,
            gfeats_ptr
    );*/

    gather_all_input_major_csr_template<2, 2, 16>
        <<<DIV_UP(nnz, 2), dim3(2, 16, 2)>>>(
            nnz,
            k_vol, 
            in_channel,
            in_feats_ptr,
            in_csr_ptr, 
            in_map_ptr,
            gfeats_ptr
    );

    /*gather_all_input_major_csr_balance<64, 8, 16>
        <<<DIV_UP(sum_nnz, 64), dim3(8, 16)>>>(
            nnz,
            k_vol, 
            sum_nnz,
            in_channel,
            in_feats_ptr,
            in_csr_ptr,  
            in_map_ptr,
            gfeats_ptr
    );*/

    // loop over all kernel offsets
    int cur_idx = 0;

    // printf("The GemmEx is used here.\n");
    // Suppose an odd kernel size
    for (int i = 0; i < k_vol - 1; i++){

        int cur_nnz = kernel_nnz.data_ptr<int>()[i];
        
        // TODO: put the zero check into the scheduler
        if (cur_nnz == 0){continue;}

        int weight_id = i < k_vol / 2 ? i : i + 1;

        // cublas GEMM for matmul
        if (TensorCoreMode){
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, cur_nnz, in_channel, 
                    &alpha, 
                    &weight_ptr[weight_id * in_channel * out_channel], CUDA_R_32F, out_channel, 
                    &gfeats_ptr[cur_idx * in_channel], CUDA_R_32F, in_channel, 
                    &beta, 
                    &sfeats_ptr[cur_idx * out_channel], CUDA_R_32F, out_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else{
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, cur_nnz, in_channel, 
                    &alpha, 
                    &weight_ptr[weight_id * in_channel * out_channel], out_channel, 
                    &gfeats_ptr[cur_idx * in_channel], in_channel, 
                    &beta, 
                    &sfeats_ptr[cur_idx * out_channel], out_channel));
        }

        cur_idx += cur_nnz;
    }

    // size_t const block_s = out_channel > PAR_THREAD ? out_channel : PAR_THREAD;
    // size_t const grid_s = (nnz * (out_channel) + block_s - 1) / block_s;
        
    /*scatter_all_output_major<<<grid_s, block_s>>>(
            nnz,
            k_vol, 
            sum_nnz,
            kernel_pos_ptr, 
            out_channel,
            sfeats_ptr, 
            out_map_ptr,
            out_feats_ptr
    );*/

    scatter_all_output_major_csr_template2<4, 2, 64>
        <<<DIV_UP(nnz, 4), dim3(64, 2, 4)>>>(
            nnz, k_vol, out_channel, sfeats_ptr, 
            out_csr_ptr, out_map_ptr, out_feats_ptr);

    /*scatter_all_output_major_csr_predecoding<4, 2, 32>
        <<<DIV_UP(nnz, 4), dim3(32, 2, 4)>>>(
            nnz, k_vol, sum_nnz, out_channel, sfeats_ptr, 
            out_csr_ptr, out_map_ptr, out_feats_ptr);*/
    
    /*scatter_all_output_major_csr_balance<128, 4, 16>
        <<<DIV_UP(sum_nnz, 128), dim3(16, 4)>>>(
            nnz, k_vol, sum_nnz, out_channel, sfeats_ptr, 
            out_csr_ptr, out_map_ptr, out_feats_ptr);*/
}


void ConvolutionBackward(const at::Tensor out_feats_grad, 
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int k_size,
                        at::Tensor in_feats_grad, 
                        at::Tensor kernel_grad, 
                        const at::Tensor kernel_nnz, 
                        const at::Tensor kernel_pos,
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        at::Tensor in_buffer,      // buffer with the shape of (sum_nnz, c_in)
                        at::Tensor out_buffer,     // buffer with the shape of (sum_nnz, c_out)  
                        const bool TensorCoreMode
                        ){

    int nnz = in_feats.size(0);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    int k_vol = k_size * k_size * k_size;

    float *ofeats_grad_ptr = out_feats_grad.data_ptr<float>();
    float *in_feats_ptr = in_feats.data_ptr<float>();
    float *weight_ptr = kernel.data_ptr<float>();
    
    float *ifeats_grad_ptr = in_feats_grad.data_ptr<float>();
    float *weight_grad_ptr = kernel_grad.data_ptr<float>();

    int *in_map_ptr = in_map.data_ptr<int>();
    int *out_map_ptr = out_map.data_ptr<int>();

    int *kernel_pos_ptr = kernel_pos.data_ptr<int>();
    
    float *in_buffer_ptr = in_buffer.data_ptr<float>();
    float *out_buffer_ptr = out_buffer.data_ptr<float>();

    int sum_nnz = kernel_nnz.sum().item<int>();

    // separate computation for center weight w[0, 0, 0]
    // cublas
    const float alpha = 1.0;
    const float beta = 0.0;
 
    cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // stride = 1
    int mid_nnz = nnz;

    // computation for w[0, 0, 0]
    if (TensorCoreMode){

        CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
        
        // W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
        CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, mid_nnz, out_channel, 
                    &alpha, 
                    &weight_ptr[k_vol / 2 * in_channel * out_channel], CUDA_R_32F, out_channel, 
                    ofeats_grad_ptr, CUDA_R_32F, out_channel, 
                    &beta, 
                    ifeats_grad_ptr, CUDA_R_32F, in_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
        CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, mid_nnz, 
                    &alpha, 
                    ofeats_grad_ptr, CUDA_R_32F, out_channel, 
                    in_feats_ptr, CUDA_R_32F, in_channel, 
                    &beta, 
                    &weight_grad_ptr[k_vol / 2 * in_channel * out_channel], CUDA_R_32F, out_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    else{

        // W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
        CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, mid_nnz, out_channel,  
                    &alpha, 
                    &weight_ptr[k_vol / 2 * in_channel * out_channel], 
                    out_channel, 
                    ofeats_grad_ptr, 
                    out_channel, 
                    &beta, 
                    ifeats_grad_ptr, 
                    in_channel));
        
        // {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
        CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, mid_nnz, 
                    &alpha, 
                    ofeats_grad_ptr, 
                    out_channel, 
                    in_feats_ptr, 
                    in_channel, 
                    &beta, 
                    &weight_grad_ptr[k_vol / 2 * in_channel * out_channel], 
                    out_channel));

    }

    // gather for ofeats_grad
    size_t const block_g = out_channel > PAR_THREAD ? out_channel : PAR_THREAD;
    size_t const grid_g = ((nnz) * (out_channel) + block_g - 1) / block_g;

    gather_all_input_major<<<grid_g, block_g>>>(
            nnz,
            k_vol, 
            sum_nnz,
            kernel_pos_ptr, 
            out_channel,
            ofeats_grad_ptr,
            out_map_ptr,
            out_buffer_ptr
    );

    // loop over all kernel offsets: 
    // W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
    int cur_idx = 0;

    for (int i = 0; i < k_vol - 1; i++){

        int cur_nnz = kernel_nnz[i].item<int>();

        // TODO: put the zero check into the scheduler
        if (cur_nnz == 0){continue;}

        int weight_id = i < k_vol / 2 ? i : i + 1;

        // cublas GEMM for matmul
        if (TensorCoreMode){
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, cur_nnz, out_channel, 
                    &alpha, 
                    &weight_ptr[weight_id * in_channel * out_channel], CUDA_R_32F, out_channel, 
                    &out_buffer_ptr[cur_idx * out_channel], CUDA_R_32F, out_channel, 
                    &beta, 
                    &in_buffer_ptr[cur_idx * in_channel], CUDA_R_32F, in_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else{
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, cur_nnz, out_channel,
                    &alpha, 
                    &weight_ptr[weight_id * in_channel * out_channel], out_channel, 
                    &out_buffer_ptr[cur_idx * out_channel], out_channel, 
                    &beta, 
                    &in_buffer_ptr[cur_idx * in_channel], in_channel));
        }
        cur_idx += cur_nnz;
    }

    // scatter for ifeats_grad
    size_t const block_s = in_channel > PAR_THREAD ? out_channel : PAR_THREAD;
    size_t const grid_s = (nnz * (in_channel) + block_s - 1) / block_s;
        
    scatter_all_output_major<<<grid_s, block_s>>>(
            nnz,
            k_vol, 
            sum_nnz,
            kernel_pos_ptr, 
            in_channel,
            in_buffer_ptr, 
            in_map_ptr,
            ifeats_grad_ptr
    );

    // gather for in_feats

    gather_all_input_major<<<grid_s, block_s>>>(
            nnz,
            k_vol, 
            sum_nnz,
            kernel_pos_ptr, 
            in_channel,
            in_feats_ptr,
            in_map_ptr,
            in_buffer_ptr
    );

    // loop over all kernel offsets: 
    // {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
    // reset current idx in the map
    cur_idx = 0;

    for (int i = 0; i < k_vol - 1; i++){

        int cur_nnz = kernel_nnz[i].item<int>();

        // TODO: put the zero check into the scheduler
        if (cur_nnz == 0){continue;}

        int weight_id = i < k_vol / 2 ? i : i + 1;

        // cublas GEMM for matmul
        if (TensorCoreMode){
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, cur_nnz, 
                    &alpha, 
                    &out_buffer_ptr[cur_idx * out_channel], CUDA_R_32F, out_channel, 
                    &in_buffer_ptr[cur_idx * in_channel], CUDA_R_32F, in_channel, 
                    &beta, 
                    &weight_grad_ptr[weight_id * in_channel * out_channel], CUDA_R_32F, out_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else{
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, cur_nnz, 
                    &alpha, 
                    &out_buffer_ptr[cur_idx * out_channel], out_channel, 
                    &in_buffer_ptr[cur_idx * in_channel], in_channel, 
                    &beta, 
                    &weight_grad_ptr[weight_id * in_channel * out_channel], out_channel));
        }
        cur_idx += cur_nnz;
    }

    return;
}