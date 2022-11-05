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

#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "cublas_utils.h"

using namespace std;

#define PAR_THREAD 256
#define DIV_UP(x, y) (x + y - 1) / y


extern "C"

void ConvolutionForward(const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int ksize_code, 
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
                        const bool TensorCore16Fast
                        ){
    
    // printf("[SubmanifoldSparseConv] - Starts.\n");

    int in_nnz = in_feats.size(0);
    int out_nnz = out_feats.size(0);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    int k_vol = kernel.size(0);

    bool data_type_half = in_feats.scalar_type() == at::ScalarType::Half;
   
    int *in_map_ptr = in_map.data_ptr<int>();
    int *out_map_ptr = out_map.data_ptr<int>();
    int *in_csr_ptr = in_csr.data_ptr<int>();
    int *out_csr_ptr = out_csr.data_ptr<int>();
    int *kpos_ptr = kernel_pos.data_ptr<int>();

    // int sum_nnz = in_buffer.size(0);
    int buffer_offset = sum_nnz * in_channel;
    // printf("sum nnz: %d", sum_nnz);

    int ksx = ksize_code / 311;
    int ksy = (ksize_code - ksx * 311) / 17;
    int ksz = ksize_code - ksx * 311 - ksy * 17;
    int mid_weight_id = (ksx - 1) / 2 * ksy * ksz + 
        (ksy - 1) / 2 * ksz + (ksz - 1) / 2;

    // cublas
    const float alpha = 1.0;
    const float beta = 0.0;
    at::Tensor alpha_half = torch::ones({1}, dtype(at::ScalarType::Half));
    at::Tensor beta_half = torch::zeros({1}, dtype(at::ScalarType::Half));

    cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();

    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

    cublasComputeType_t ComputeType;
    cudaDataType_t DataType;
    if (data_type_half){
        ComputeType = CUBLAS_COMPUTE_16F;
        DataType = CUDA_R_16F;
    }
    else{
        ComputeType = TensorCore16Fast ? 
            CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_32F_FAST_TF32;
        DataType = CUDA_R_32F;
    }

    /********************************************************************/
    // default stream: gather

    if (data_type_half){
        gather_all_input_major_csr_half<<<DIV_UP(in_nnz, 4), 
            dim3(DIV_UP(in_channel, 8), 2, 4), 0, 0>>>(
                in_nnz, in_channel, reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()), 
                kpos_ptr, in_csr_ptr, in_map_ptr, reinterpret_cast<half *>(buffer.data_ptr<at::Half>())
        );
    }
    else{
        gather_all_input_major_csr_float<<<DIV_UP(in_nnz, 4), 
            dim3(DIV_UP(in_channel, 4), 2, 4), 0, 0>>>(
                in_nnz, in_channel, in_feats.data_ptr<float>(), kpos_ptr, 
                in_csr_ptr, in_map_ptr, buffer.data_ptr<float>()
        );
    }

    /********************************************************************/
    // create the streams
    int n_stream = 4;

    cudaStream_t *pl_stream;
    pl_stream = (cudaStream_t *)new cudaStream_t[n_stream];
    
    for (int i = 0; i < n_stream; i++) {
        cudaStreamCreateWithFlags(&pl_stream[i], cudaStreamDefault);
    }

    /********************************************************************/
    // loop over all kernel offsets
    int cur_idx = 0;

    // printf("The GemmEx is used here.\n");
    // Suppose an odd kernel size
    for (int i = 0; i < k_vol; i++){

        int cur_nnz = kernel_nnz.data_ptr<int>()[i];
        
        // TODO: put the zero check into the scheduler
        if (cur_nnz == 0){continue;}

        int stream_id = i % n_stream;

        CUBLAS_CHECK(cublasSetStream(cublasH, pl_stream[stream_id]));

        if (data_type_half){
            // cublas GEMM for matmul
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, cur_nnz, in_channel, 
                    (reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>())),
                    // &weight_ptr[i * in_channel * out_channel], 
                    (reinterpret_cast<half *>(kernel.data_ptr<at::Half>()
                        + i * in_channel * out_channel)),
                    DataType, out_channel, 
                    // &buf_ptr[cur_idx * in_channel], 
                    (reinterpret_cast<half *>(buffer.data_ptr<at::Half>()
                        + cur_idx * in_channel)),
                    DataType, in_channel, 
                    (reinterpret_cast<half *>(beta_half.data_ptr<at::Half>())),  
                    // &buf_ptr[buffer_offset + cur_idx * out_channel], 
                    (reinterpret_cast<half *>(buffer.data_ptr<at::Half>()
                        + buffer_offset + cur_idx * out_channel)), 
                    DataType, out_channel,
                    ComputeType,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            
            /*CUBLAS_CHECK(cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, cur_nnz, in_channel, 
                    (reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>())), 
                    // &weight_ptr[i * in_channel * out_channel], 
                    (reinterpret_cast<half *>(kernel.data_ptr<at::Half>()
                        + i * in_channel * out_channel)),
                    out_channel, 
                    // &buf_ptr[cur_idx * in_channel], 
                    (reinterpret_cast<half *>(buffer.data_ptr<at::Half>()
                        + cur_idx * in_channel)),
                    in_channel, 
                    (reinterpret_cast<half *>(beta_half.data_ptr<at::Half>())), 
                    // &buf_ptr[buffer_offset + cur_idx * out_channel], 
                    (reinterpret_cast<half *>(buffer.data_ptr<at::Half>()
                        + buffer_offset + cur_idx * out_channel)), 
                    out_channel));*/
        }
        else{
            // cublas GEMM for matmul
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, cur_nnz, in_channel, 
                    &alpha, 
                    // &weight_ptr[i * in_channel * out_channel], 
                    (kernel.data_ptr<float>() + i * in_channel * out_channel),
                    DataType, out_channel, 
                    // &buf_ptr[cur_idx * in_channel], 
                    (buffer.data_ptr<float>() + cur_idx * in_channel),
                    DataType, in_channel, 
                    &beta, 
                    // &buf_ptr[buffer_offset + cur_idx * out_channel], 
                    (buffer.data_ptr<float>() + buffer_offset + cur_idx * out_channel), 
                    DataType, out_channel,
                    ComputeType,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }

        cur_idx += cur_nnz;
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < n_stream; i++) {
        cudaStreamDestroy(pl_stream[i]);
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

    /*scatter_all_output_major_csr_template2<4, 2, 64>
        <<<DIV_UP(nnz, 4), dim3(64, 2, 4), 0, 0>>>(
            nnz, k_vol, out_channel, &buf_ptr[buffer_offset], kpos_ptr, 
            out_csr_ptr, out_map_ptr, out_feats_ptr);*/

    /*scatter_all_output_major_csr_predecoding<4, 2, 32>
        <<<DIV_UP(nnz, 4), dim3(32, 2, 4)>>>(
            nnz, k_vol, sum_nnz, out_channel, sfeats_ptr, 
            out_csr_ptr, out_map_ptr, out_feats_ptr);*/
    
    /*scatter_all_output_major_csr_balance<128, 4, 16>
        <<<DIV_UP(sum_nnz, 128), dim3(16, 4)>>>(
            nnz, k_vol, sum_nnz, out_channel, sfeats_ptr, 
            out_csr_ptr, out_map_ptr, out_feats_ptr);*/

    /********************************************************************/
    // default stream
    
    if (data_type_half){
        scatter_all_output_major_csr_half<<<DIV_UP(out_nnz, 4), 
            dim3(DIV_UP(out_channel, 8), 4), 0, 0>>>(
                out_nnz, out_channel, (reinterpret_cast<half *>(buffer.data_ptr<at::Half>() + buffer_offset)), 
                kpos_ptr, out_csr_ptr, out_map_ptr, reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()));
    }
    else{
        scatter_all_output_major_csr_float<<<DIV_UP(out_nnz, 4), 
            dim3(DIV_UP(out_channel, 4), 4), 0, 0>>>(
                out_nnz, out_channel, (buffer.data_ptr<float>() + buffer_offset), kpos_ptr, 
                out_csr_ptr, out_map_ptr, out_feats.data_ptr<float>());
    }


    // computation for w[0, 0, 0]
    // in_nnz == out_nnz
    if (separate_mid){
        CUBLAS_CHECK(cublasSetStream(cublasH, 0));

        if (data_type_half){
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, in_nnz, in_channel, 
                    reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                    // &weight_ptr[mid_weight_id * in_channel * out_channel], 
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>() 
                        + mid_weight_id * in_channel * out_channel),
                    DataType, out_channel, 
                    reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()), 
                    DataType, in_channel, 
                    reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                    reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()), 
                    DataType, out_channel,
                    ComputeType,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            
            /*CUBLAS_CHECK(cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, in_nnz, in_channel, 
                    reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                    // &weight_ptr[mid_weight_id * in_channel * out_channel], 
                    reinterpret_cast<half *>(kernel.data_ptr<at::Half>() 
                        + mid_weight_id * in_channel * out_channel),
                    out_channel, 
                    reinterpret_cast<half *>(in_feats.data_ptr<at::Half>()), 
                    in_channel, 
                    reinterpret_cast<half *>(alpha_half.data_ptr<at::Half>()), 
                    reinterpret_cast<half *>(out_feats.data_ptr<at::Half>()), 
                    out_channel));*/
        }
        else{
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 
                    out_channel, in_nnz, in_channel, 
                    &alpha, 
                    // &weight_ptr[mid_weight_id * in_channel * out_channel], 
                    (kernel.data_ptr<float>() + mid_weight_id * in_channel * out_channel),
                    DataType, out_channel, 
                    in_feats.data_ptr<float>(), 
                    DataType, in_channel, 
                    &alpha, 
                    out_feats.data_ptr<float>(), 
                    DataType, out_channel,
                    ComputeType,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

}


void ConvolutionBackward(const at::Tensor out_feats_grad, 
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int ksize_code,
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

    int innz = in_feats.size(0);
    int onnz = out_feats_grad.size(0);
    bool separate_mid = (innz == onnz);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    int k_vol = kernel.size(0);

    float *ofeats_grad_ptr = out_feats_grad.data_ptr<float>();
    float *in_feats_ptr = in_feats.data_ptr<float>();
    float *weight_ptr = kernel.data_ptr<float>();
    
    float *ifeats_grad_ptr = in_feats_grad.data_ptr<float>();
    float *weight_grad_ptr = kernel_grad.data_ptr<float>();

    int *in_map_ptr = in_map.data_ptr<int>();
    int *out_map_ptr = out_map.data_ptr<int>();
    int *in_csr_ptr = in_csr.data_ptr<int>();
    int *out_csr_ptr = out_csr.data_ptr<int>();

    int *kpos_ptr = kernel_pos.data_ptr<int>();

    int ksx = ksize_code / 311;
    int ksy = (ksize_code - ksx * 311) / 17;
    int ksz = ksize_code - ksx * 311 - ksy * 17;
    int mid_weight_id = (ksx - 1) / 2 * ksy * ksz + 
        (ksy - 1) / 2 * ksz + (ksz - 1) / 2;
    
    float *buf_ptr = buffer.data_ptr<float>();

    int buffer_offset = sum_nnz * in_channel;

    // cublas
    const float alpha = 1.0;
    const float beta = 0.0;
 
    cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // gather for ofeats_grad
    gather_all_input_major_csr_float<<<DIV_UP(onnz, 4), 
        dim3(DIV_UP(out_channel, 4), 2, 4), 0, 0>>>(
            onnz, // in_nnz,
            out_channel, // in_channel,
            ofeats_grad_ptr, // in_feats_ptr,
            kpos_ptr, 
            out_csr_ptr, // in_csr_ptr, 
            out_map_ptr, // in_map_ptr,
            &buf_ptr[buffer_offset] // buf_ptr
    );

    /*size_t const block_g = out_channel > PAR_THREAD ? out_channel : PAR_THREAD;
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
    );*/

    // loop over all kernel offsets: 
    // W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
    int cur_idx = 0;

    for (int i = 0; i < k_vol; i++){

        int cur_nnz = kernel_nnz.data_ptr<int>()[i];
        
        // TODO: put the zero check into the scheduler
        if (cur_nnz == 0){continue;}

        // cublas GEMM for matmul
        if (TensorCoreMode){
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, cur_nnz, out_channel, 
                    &alpha, 
                    &weight_ptr[i * in_channel * out_channel], CUDA_R_32F, out_channel, 
                    &buf_ptr[buffer_offset + cur_idx * out_channel], CUDA_R_32F, out_channel, 
                    &beta, 
                    &buf_ptr[cur_idx * in_channel], CUDA_R_32F, in_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else{
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, cur_nnz, out_channel,
                    &alpha, 
                    &weight_ptr[i * in_channel * out_channel], out_channel, 
                    &buf_ptr[buffer_offset + cur_idx * out_channel], out_channel, 
                    &beta, 
                    &buf_ptr[cur_idx * in_channel], in_channel));
        }
        cur_idx += cur_nnz;
    }

    // scatter for ifeats_grad

    scatter_all_output_major_csr_float<<<DIV_UP(innz, 4), 
        dim3(DIV_UP(in_channel, 4), 4), 0, 0>>>(
            innz, // out_nnz, 
            in_channel, // out_channel, 
            buf_ptr, // &buf_ptr[buffer_offset], 
            kpos_ptr, 
            in_csr_ptr, // out_csr_ptr, 
            in_map_ptr, // out_map_ptr, 
            ifeats_grad_ptr // out_feats_ptr
    );

    /*size_t const block_s = in_channel > PAR_THREAD ? out_channel : PAR_THREAD;
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
    );*/

    // gather for in_feats
    gather_all_input_major_csr_float<<<DIV_UP(innz, 4), 
        dim3(DIV_UP(in_channel, 4), 2, 4), 0, 0>>>(
            innz,
            in_channel,
            in_feats_ptr,
            kpos_ptr, 
            in_csr_ptr, 
            in_map_ptr,
            buf_ptr
    );
    /*gather_all_input_major<<<grid_s, block_s>>>(
            nnz,
            k_vol, 
            sum_nnz,
            kernel_pos_ptr, 
            in_channel,
            in_feats_ptr,
            in_map_ptr,
            in_buffer_ptr
    );*/

    // loop over all kernel offsets: 
    // {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
    // reset current idx in the map
    cur_idx = 0;

    for (int i = 0; i < k_vol; i++){

        int cur_nnz = kernel_nnz.data_ptr<int>()[i];
        
        // TODO: put the zero check into the scheduler
        if (cur_nnz == 0){continue;}

        // cublas GEMM for matmul
        if (TensorCoreMode){
            CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, cur_nnz, 
                    &alpha, 
                    &buf_ptr[buffer_offset + cur_idx * out_channel], CUDA_R_32F, out_channel, 
                    &buf_ptr[cur_idx * in_channel], CUDA_R_32F, in_channel, 
                    &beta, 
                    &weight_grad_ptr[i * in_channel * out_channel], CUDA_R_32F, out_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else{
            CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, cur_nnz, 
                    &alpha, 
                    &buf_ptr[buffer_offset + cur_idx * out_channel], out_channel, 
                    &buf_ptr[cur_idx * in_channel], in_channel, 
                    &beta, 
                    &weight_grad_ptr[i * in_channel * out_channel], out_channel));
        }
        cur_idx += cur_nnz;
    }

    // separate computation for center weight w[0, 0, 0]
    // computation for w[0, 0, 0]
    if (separate_mid){
        if (TensorCoreMode){

        CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
        
        // W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
        CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, innz, out_channel, 
                    &alpha, 
                    &weight_ptr[mid_weight_id * in_channel * out_channel], CUDA_R_32F, out_channel, 
                    ofeats_grad_ptr, CUDA_R_32F, out_channel, 
                    &alpha, 
                    ifeats_grad_ptr, CUDA_R_32F, in_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
        CUBLAS_CHECK(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, innz, 
                    &alpha, 
                    ofeats_grad_ptr, CUDA_R_32F, out_channel, 
                    in_feats_ptr, CUDA_R_32F, in_channel, 
                    &alpha, 
                    &weight_grad_ptr[mid_weight_id * in_channel * out_channel], CUDA_R_32F, out_channel,
                    CUBLAS_COMPUTE_32F_FAST_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else{

        // W^T X {\delta{out_feats}} = {\delta{in_feats}}^T
        CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    in_channel, innz, out_channel,  
                    &alpha, 
                    &weight_ptr[mid_weight_id * in_channel * out_channel], 
                    out_channel, 
                    ofeats_grad_ptr, 
                    out_channel, 
                    &alpha, 
                    ifeats_grad_ptr, 
                    in_channel));
        
        // {\delta{out_feats}}^T X in_feats = {\delta{W}}^T
        CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
                    out_channel, in_channel, innz, 
                    &alpha, 
                    ofeats_grad_ptr, 
                    out_channel, 
                    in_feats_ptr, 
                    in_channel, 
                    &alpha, 
                    &weight_grad_ptr[mid_weight_id * in_channel * out_channel], 
                    out_channel));

        }
    }

    return;
}
