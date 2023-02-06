#include "hash.h"
#include "hash.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <torch/extension.h>

#define DIV_UP(x, y) (x + y - 1) / y

at::Tensor HashMap(
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
                at::Tensor kernel_kpos,
                at::Tensor kernel_qkpos,
                const bool separate_mid
                ){
  
    int in_nnz = in_coords.size(0);
    int table_size = 2 * pow(2, ceil(log2((double)in_nnz)));

    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *imap_ptr = imap.data_ptr<int>();
    int *omap_ptr = omap.data_ptr<int>();
    int *knnz_ptr = kernel_nnz.data_ptr<int>();
    int *kpos_ptr = kernel_kpos.data_ptr<int>();
    int *qkpos_ptr = kernel_qkpos.data_ptr<int>();
    int *icsr_ptr = icsr.data_ptr<int>();
    int *ocsr_ptr = ocsr.data_ptr<int>();

    // stride decoding
    int l_stride_x = l_stride_code / 311;
    int l_stride_y = (l_stride_code - l_stride_x * 311) / 17;
    int l_stride_z = l_stride_code - l_stride_x * 311 - l_stride_y * 17;

    int t_stride_x = t_stride_code / 311;
    int t_stride_y = (t_stride_code - t_stride_x * 311) / 17;
    int t_stride_z = t_stride_code - t_stride_x * 311 - t_stride_y * 17;
 
    int stride_x = l_stride_x * t_stride_x;
    int stride_y = l_stride_y * t_stride_y;
    int stride_z = l_stride_z * t_stride_z;

    int k_size_x = k_size_code / 311;
    int k_size_y = (k_size_code - k_size_x * 311) / 17;
    int k_size_z = k_size_code - k_size_x * 311 - k_size_y * 17;

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
    int n_stream = 2;

    cudaStream_t *pl_stream;
    pl_stream = (cudaStream_t *)new cudaStream_t[n_stream];
    
    for (int i = 0; i < n_stream; i++) {
        cudaStreamCreateWithFlags(&pl_stream[i], cudaStreamDefault);
    }

    /********************************************************************/
    // default stream
    int out_nnz;
    at::Tensor out_coords;

    if (separate_mid){
        // TODO: check if new allocation occurs
        out_coords = in_coords;
        out_nnz = in_nnz;
    }
    else{
        if ((l_stride_x == 1 || l_stride_x == k_size_x) && 
            (l_stride_y == 1 || l_stride_y == k_size_y) && 
            (l_stride_z == 1 || l_stride_z == k_size_z)){
        
        at::Tensor ocoords_code_space = torch::zeros({in_nnz},   
            at::device(in_coords.device()).dtype(at::ScalarType::Long));

        int64_t *ocoords_code_ptr = ocoords_code_space.data_ptr<int64_t>(); 

        coordsDownsample<3259><<<DIV_UP(in_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            in_nnz, stride_x, stride_y, stride_z, in_coords_ptr, ocoords_code_ptr
        );

        // in defaut order: b -> x -> y -> z
        thrust::sort(thrust::cuda::par.on(0), 
            ocoords_code_ptr, ocoords_code_ptr + in_nnz);

        int64_t *new_end = thrust::unique(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + in_nnz);
        
        out_nnz = new_end - ocoords_code_ptr;

        out_coords = torch::zeros({out_nnz, 4}, 
            at::device(in_coords.device()).dtype(at::ScalarType::Int));
        
        coordsGenerator<3259><<<DIV_UP(out_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            out_nnz, ocoords_code_ptr, out_coords.data_ptr<int>()
        );

        }
        else{
        at::Tensor ocoords_code_space = - torch::ones({in_nnz * k_vol},   
            at::device(in_coords.device()).dtype(at::ScalarType::Long));

        int64_t *ocoords_code_ptr = ocoords_code_space.data_ptr<int64_t>(); 

        // coordsDownsample<3041><<<DIV_UP(in_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
        //    in_nnz, stride_x, stride_y, stride_z, in_coords_ptr, ocoords_code_ptr
        //);
        coordsDownsampleExpand<3259, 4><<<DIV_UP(in_nnz, 4), dim3(4, 4, 1), 0, 0>>>(
            in_nnz, k_vol, k_size_x, k_size_y, k_size_z, t_stride_x, t_stride_y, t_stride_z, 
            stride_x, stride_y, stride_z, in_coords_ptr, ocoords_code_ptr
        );

        // extract the valid output coords code
        int64_t *valid_end = thrust::remove(thrust::cuda::par.on(0), 
            ocoords_code_ptr, ocoords_code_ptr + (in_nnz * k_vol), -1);

        int valid_num = valid_end - ocoords_code_ptr;
        
        // in defaut order: b -> x -> y -> z
        thrust::sort(thrust::cuda::par.on(0), 
            ocoords_code_ptr, ocoords_code_ptr + valid_num);

        int64_t *new_end = thrust::unique(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + valid_num);
        
        out_nnz = new_end - ocoords_code_ptr;

        out_coords = torch::zeros({out_nnz, 4}, 
            at::device(in_coords.device()).dtype(at::ScalarType::Int));
        
        coordsGenerator<3259><<<DIV_UP(out_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            out_nnz, ocoords_code_ptr, out_coords.data_ptr<int>()
        );
        }
    }

    int *out_coords_ptr = out_coords.data_ptr<int>();
    
    // build the input coords hash table for query
    insertHash<<<DIV_UP(in_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
        in_nnz, table_size, in_coords_ptr, index_ptr
    );
    
    insertVal<<<DIV_UP(table_size, 16), dim3(16, 1, 1), 0, 0>>>(
        in_nnz, table_size, in_coords_ptr, index_ptr, value_ptr
    );

    // query input id from output id 
    queryHash_wholemap<16, 4><<<DIV_UP(out_nnz, 16), dim3(4, 16), 0, 0>>>(
        in_nnz, out_nnz, table_size, out_coords_ptr, k_size_x, k_size_y, k_size_z, k_vol, 
        t_stride_x, t_stride_y, t_stride_z, value_ptr, 
        index_ptr, imap_ptr, omap_ptr, knnz_ptr, separate_mid
    );

    exclusive_scan_for_kernel_quantified<<<1, k_vol, 0, 0>>>(
        k_vol + 1, knnz_ptr, 128, kpos_ptr, qkpos_ptr
    );

    /********************************************************************/
    // created stream 0

    mapping_counter<<<DIV_UP(in_nnz, 16), dim3(16, 1, 1), 0, pl_stream[0]>>>(
       in_nnz, k_vol, imap_ptr, icsr_ptr
    );

    /********************************************************************/
    // created stream 1

    mapping_counter<<<DIV_UP(out_nnz, 16), dim3(16, 1, 1), 0, pl_stream[1]>>>(
        out_nnz, k_vol, omap_ptr, ocsr_ptr
    );

    /********************************************************************/
    // created stream 0

    thrust::remove(thrust::cuda::par.on(pl_stream[0]), 
        imap_ptr, imap_ptr + in_nnz * k_vol, -1);

    /********************************************************************/
    // created stream 1

    thrust::remove(thrust::cuda::par.on(pl_stream[1]), 
        omap_ptr, omap_ptr + out_nnz * k_vol, -1);

    /********************************************************************/
    // created stream 0

    thrust::exclusive_scan(thrust::cuda::par.on(pl_stream[0]), 
        icsr_ptr, &icsr_ptr[in_nnz + 1], icsr_ptr);

    /********************************************************************/
    // created stream 1

    thrust::exclusive_scan(thrust::cuda::par.on(pl_stream[1]),  
        ocsr_ptr, &ocsr_ptr[out_nnz + 1], ocsr_ptr);

    /********************************************************************/

    // int sum_nnz = kernel_pos[k_vol - 1].item<int>();

    cudaDeviceSynchronize();
    for (int i = 0; i < n_stream; i++) {
        cudaStreamDestroy(pl_stream[i]);
    }

    return out_coords;
}


at::Tensor HashMap_simple(
                const at::Tensor in_coords, 
                const int batch_size, 
                const int k_size_code, 
                const int k_vol, 
                const int c_in, 
                const int c_out, 
                const int l_stride_code, 
                const int t_stride_code, 
                at::Tensor map,
                at::Tensor kernel_nnz,
                at::Tensor kernel_pos,
                at::Tensor kernel_kpos,
                const bool separate_mid
                ){
    
    int in_nnz = in_coords.size(0);
    int table_size = 2 * pow(2, ceil(log2((double)(in_nnz))));
    // printf("table size: %d", table_size);

    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *map_ptr = map.data_ptr<int>();
    int *knnz_ptr = kernel_nnz.data_ptr<int>();
    int *kpos_ptr = kernel_pos.data_ptr<int>();
    int *qkpos_ptr = kernel_kpos.data_ptr<int>();

    // stride decoding
    int l_stride_x = l_stride_code / 311;
    int l_stride_y = (l_stride_code - l_stride_x * 311) / 17;
    int l_stride_z = l_stride_code - l_stride_x * 311 - l_stride_y * 17;

    int t_stride_x = t_stride_code / 311;
    int t_stride_y = (t_stride_code - t_stride_x * 311) / 17;
    int t_stride_z = t_stride_code - t_stride_x * 311 - t_stride_y * 17;
 
    int stride_x = l_stride_x * t_stride_x;
    int stride_y = l_stride_y * t_stride_y;
    int stride_z = l_stride_z * t_stride_z;

    int k_size_x = k_size_code / 311;
    int k_size_y = (k_size_code - k_size_x * 311) / 17;
    int k_size_z = k_size_code - k_size_x * 311 - k_size_y * 17;

    /********************************************************************/
    // default stream
    at::Tensor index = - torch::ones({table_size}, 
        at::device(in_coords.device()).dtype(at::ScalarType::Int));
    at::Tensor value = torch::zeros({table_size}, 
        at::device(in_coords.device()).dtype(at::ScalarType::Long));

    int *index_ptr = index.data_ptr<int>();
    uint64_t *value_ptr = (uint64_t *)(value.data_ptr<int64_t>());

    /********************************************************************/
    // default stream
    int out_nnz;
    at::Tensor out_coords;

    if (separate_mid){
        // TODO: check if new allocation occurs
        out_coords = in_coords;
        out_nnz = in_nnz;
    }
    else{
        if ((l_stride_x == 1 || l_stride_x == k_size_x) && 
            (l_stride_y == 1 || l_stride_y == k_size_y) && 
            (l_stride_z == 1 || l_stride_z == k_size_z)){
        
        at::Tensor ocoords_code_space = torch::zeros({in_nnz},   
            at::device(in_coords.device()).dtype(at::ScalarType::Long));

        int64_t *ocoords_code_ptr = ocoords_code_space.data_ptr<int64_t>(); 

        coordsDownsample<3259><<<DIV_UP(in_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            in_nnz, stride_x, stride_y, stride_z, in_coords_ptr, ocoords_code_ptr
        );

        // in defaut order: b -> x -> y -> z
        thrust::sort(thrust::cuda::par.on(0), 
            ocoords_code_ptr, ocoords_code_ptr + in_nnz);

        int64_t *new_end = thrust::unique(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + in_nnz);
        
        out_nnz = new_end - ocoords_code_ptr;

        out_coords = torch::zeros({out_nnz, 4}, 
            at::device(in_coords.device()).dtype(at::ScalarType::Int));
        
        coordsGenerator<3259><<<DIV_UP(out_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            out_nnz, ocoords_code_ptr, out_coords.data_ptr<int>()
        );

        }
        else{
        at::Tensor ocoords_code_space = - torch::ones({in_nnz * k_vol},   
            at::device(in_coords.device()).dtype(at::ScalarType::Long));

        int64_t *ocoords_code_ptr = ocoords_code_space.data_ptr<int64_t>(); 

        // coordsDownsample<3041><<<DIV_UP(in_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
        //    in_nnz, stride_x, stride_y, stride_z, in_coords_ptr, ocoords_code_ptr
        //);
        coordsDownsampleExpand<3259, 4><<<DIV_UP(in_nnz, 4), dim3(4, 4), 0, 0>>>(
            in_nnz, k_vol, k_size_x, k_size_y, k_size_z, t_stride_x, t_stride_y, t_stride_z, 
            stride_x, stride_y, stride_z, in_coords_ptr, ocoords_code_ptr
        );

        // extract the valid output coords code
        int64_t *valid_end = thrust::remove(thrust::cuda::par.on(0), 
            ocoords_code_ptr, ocoords_code_ptr + (in_nnz * k_vol), -1);

        int valid_num = valid_end - ocoords_code_ptr;

        // in defaut order: b -> x -> y -> z
        thrust::sort(thrust::cuda::par.on(0), 
            ocoords_code_ptr, ocoords_code_ptr + valid_num);

        int64_t *new_end = thrust::unique(thrust::cuda::par.on(0),
            ocoords_code_ptr, ocoords_code_ptr + valid_num);
        
        out_nnz = new_end - ocoords_code_ptr;

        out_coords = torch::zeros({out_nnz, 4}, 
            at::device(in_coords.device()).dtype(at::ScalarType::Int));
        
        coordsGenerator<3259><<<DIV_UP(out_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
            out_nnz, ocoords_code_ptr, out_coords.data_ptr<int>()
        );
        }
    }

    int *out_coords_ptr = out_coords.data_ptr<int>();

    // build the input coords hash table for query
    insertHash<<<DIV_UP(in_nnz, 16), dim3(16, 1, 1), 0, 0>>>(
        in_nnz, table_size, in_coords_ptr, index_ptr
    );
    
    insertVal<<<DIV_UP(table_size, 16), dim3(16, 1, 1), 0, 0>>>(
        in_nnz, table_size, in_coords_ptr, index_ptr, value_ptr
    );

    // printf("insertVal done.");

    // query input id from output id 
    queryHash_simple<16, 4><<<DIV_UP(out_nnz, 16), dim3(4, 16), 0, 0>>>(
        in_nnz, out_nnz, table_size, out_coords_ptr, k_size_x, k_size_y, k_size_z, k_vol, 
        t_stride_x, t_stride_y, t_stride_z, value_ptr, 
        index_ptr, map_ptr, knnz_ptr, separate_mid
    );

    // printf("queryHash done.");

    exclusive_scan_for_kernel_quantified<<<1, k_vol, 0, 0>>>(
        k_vol + 1, knnz_ptr, 128, kpos_ptr, qkpos_ptr
    );

    return out_coords;
}