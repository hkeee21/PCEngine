#include "hash.h"
#include "hash.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/scan.h>
#include <thrust/remove.h>

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <torch/extension.h>

#define DIV_UP(x, y) (x + y - 1) / y

using namespace std;

int HashMap(const at::Tensor in_coords, 
                const int k_size, 
                const int c_in, 
                const int c_out, 
                at::Tensor imap,
                at::Tensor omap,  
                at::Tensor icsr,
                at::Tensor ocsr,
                at::Tensor kernel_nnz, 
                at::Tensor kernel_pos
                ){

    int nnz = in_coords.size(0);
    int table_size = 2 * pow(2, ceil(log2((double)nnz)));
    int k_vol = k_size * k_size * k_size;

    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *imap_ptr = imap.data_ptr<int>();
    int *omap_ptr = omap.data_ptr<int>();
    int *kernel_nnz_ptr = kernel_nnz.data_ptr<int>();
    int *kernel_pos_ptr = kernel_pos.data_ptr<int>();
    int *icsr_ptr = icsr.data_ptr<int>();
    int *ocsr_ptr = ocsr.data_ptr<int>();

    /********************************************************************/
    // default stream

    at::Tensor index = - torch::ones({table_size}, 
        at::device(in_coords.device()).dtype(at::ScalarType::Int));
    at::Tensor value = torch::zeros({table_size}, 
        at::device(in_coords.device()).dtype(at::ScalarType::Long));

    int *index_ptr = index.data_ptr<int>();
    uint64_t *value_ptr = (uint64_t *)(value.data_ptr<int64_t>());

    /********************************************************************/
    // create the streams
    int n_stream = 5;

    cudaStream_t *pl_stream;
    pl_stream = (cudaStream_t *)new cudaStream_t[n_stream];
    
    for (int i = 0; i < n_stream; i++) {
        cudaStreamCreateWithFlags(&pl_stream[i], cudaStreamNonBlocking);
    }

    /********************************************************************/
    // created stream 0

    insertHash<<<DIV_UP(nnz, 16), dim3(16, 1, 1), 0, pl_stream[0]>>>(
        nnz, table_size, in_coords_ptr, index_ptr
    );
    
    insertVal<<<DIV_UP(table_size, 16), dim3(16, 1, 1), 0, pl_stream[0]>>>(
        nnz, table_size, in_coords_ptr, index_ptr, value_ptr
    );

    queryHash_wholemap<16, 4><<<DIV_UP(nnz, 16), dim3(16, 4), 0, pl_stream[0]>>>(
        nnz, table_size, in_coords_ptr, k_size, k_vol, value_ptr, 
        index_ptr, imap_ptr, omap_ptr, kernel_nnz_ptr
    );

    cudaEvent_t K_qH_done;
    cudaEventCreateWithFlags(&K_qH_done, cudaEventDisableTiming);
    cudaEventRecord(K_qH_done, pl_stream[0]);

    /********************************************************************/
    // default stream

    cudaStreamWaitEvent(0, K_qH_done);

    int sum_nnz = kernel_nnz.sum().item<int>();

    // cudaMallocAsync((void **)&buffer_ptr, sum_nnz * (c_in + c_out) * sizeof(float), 0);
    // at::Tensor buffer = torch::zeros({sum_nnz, (c_in + c_out)}, 
    //     at::device(in_coords.device()).dtype(at::ScalarType::Float));

    /********************************************************************/
    // created stream 0

    thrust::exclusive_scan(thrust::cuda::par.on(pl_stream[0]), 
        kernel_nnz_ptr, &kernel_nnz_ptr[k_vol - 1], kernel_pos_ptr);

    /********************************************************************/
    // created stream 1

    cudaStreamWaitEvent(pl_stream[1], K_qH_done);

    mapping_counter<<<DIV_UP(nnz, 16), dim3(16, 1, 1), 0, pl_stream[1]>>>(
        nnz, k_vol, imap_ptr, icsr_ptr
    );

    cudaEvent_t K_mci_done;
    cudaEventCreateWithFlags(&K_mci_done, cudaEventDisableTiming);
    cudaEventRecord(K_mci_done, pl_stream[1]);

    thrust::remove(thrust::cuda::par.on(pl_stream[1]), 
        imap_ptr, imap_ptr + nnz * (k_vol - 1), -1);

    /********************************************************************/
    // created stream 2

    cudaStreamWaitEvent(pl_stream[2], K_qH_done);

    mapping_counter<<<DIV_UP(nnz, 16), dim3(16, 1, 1), 0, pl_stream[2]>>>(
        nnz, k_vol, omap_ptr, ocsr_ptr
    );

    cudaEvent_t K_mco_done;
    cudaEventCreateWithFlags(&K_mco_done, cudaEventDisableTiming);
    cudaEventRecord(K_mco_done, pl_stream[2]);

    thrust::remove(thrust::cuda::par.on(pl_stream[2]), 
        omap_ptr, omap_ptr + nnz * (k_vol - 1), -1);


    /********************************************************************/
    // created stream 3

    cudaStreamWaitEvent(pl_stream[3], K_mci_done);  

    thrust::exclusive_scan(thrust::cuda::par.on(pl_stream[3]), 
        icsr_ptr, &icsr_ptr[nnz + 1], icsr_ptr);

    /********************************************************************/
    // created stream 4

    cudaStreamWaitEvent(pl_stream[4], K_mco_done);  

    thrust::exclusive_scan(thrust::cuda::par.on(pl_stream[4]),  
        ocsr_ptr, &ocsr_ptr[nnz + 1], ocsr_ptr);

    /********************************************************************/
    cudaDeviceSynchronize();
    for (int i = 0; i < n_stream; i++) {
        cudaStreamDestroy(pl_stream[i]);
    } 

    // cudaMallocAsync((void **)&buffer_ptr, sum_nnz * (c_in + c_out) * sizeof(float), 0);
    // at::Tensor buffer = torch::zeros({sum_nnz, (c_in + c_out)}, 
    //     at::device(in_coords.device()).dtype(at::ScalarType::Float));

    // at::Tensor buffer = torch::from_blob(buffer_ptr, {sum_nnz, (c_in + c_out)}, 
    //     at::device(in_coords.device()).dtype(at::ScalarType::Float));
    
    return sum_nnz;
}