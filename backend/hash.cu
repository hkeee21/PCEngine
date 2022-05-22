#include "hash.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <torch/extension.h>

using namespace std;

#define BLOCK_SIZE 32
#define COLLISION_BOUND 20

extern "C"


inline __device__ uint64_t coord_hash(const int ix, const int iy, const int iz){
    // +1 to avoid val==0
    return ((uint64_t)ix * 73856093 + (uint64_t)iy * 19349669 + (uint64_t)iz * 83492791 + 1);
}

inline __device__ uint64_t shift_hash(const int size, const uint64_t value){
    return ((value + 1) % ((uint64_t)size - 2));
}


__global__ void insertHash(const int nnz, const int size, const int *__restrict__ coord, int *idx){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // exclude illegal id number
    if (id < nnz){
        
        uint64_t temp_val = coord_hash(coord[3 * id], coord[3 * id + 1], coord[3 * id + 2]);

        // temp_val is unique
        uint64_t table_id = temp_val % (uint64_t)size;

        // cuckoo hashing

        int old_idx = atomicExch(&idx[table_id], id);
        // uint64_t old_val = atomicExch(&val[table_id], temp_val);
        // handle collision
        while(old_idx > -1){
            table_id = (table_id + 97) % size;
            old_idx = atomicExch(&idx[table_id], old_idx);
            // old_val = atomicExch(&val[table_id], old_val);
        }  
    }
}

__global__ void insertVal(const int nnz, const int size, const int *__restrict__ coord, const int *idx, uint64_t *val){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size){
        
        int input_id = idx[id];
        if (input_id < nnz && input_id > -1){
            val[id] = coord_hash(coord[3 *input_id], coord[3 * input_id + 1], coord[3 * input_id + 2]);
        }
    }
}


__global__ void queryHash(const int nnz, const int size, const int *__restrict__ coord,
                    const int ks, const int kv, const uint64_t *val, const int *idx, int *map)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord

    int offset_id = id / nnz;

    int nnz_id = id % nnz;

    // exclude illegal id number
    if (offset_id < kv)
    {
        int kx = offset_id / (ks * ks) - (ks - 1) / 2;
        int ky = (offset_id / ks) % ks - (ks - 1) / 2;
        int kz = offset_id % ks - (ks - 1) / 2;

        int colli_num = 0; 

        int Ix = coord[nnz_id * 3] + kx;
        int Iy = coord[nnz_id * 3 + 1] + ky;
        int Iz = coord[nnz_id * 3 + 2] + kz;
        
        uint64_t target_val = coord_hash(Ix, Iy, Iz);

        uint64_t target_id = target_val % (uint64_t)size;

        // find target or empty
        while (val[target_id] != target_val && idx[target_id] > -1){
            colli_num += 1;
            if (colli_num == COLLISION_BOUND){map[id] = -1; return;}
            target_id = (target_id + 97) % size;
        }
        // set map = input id or -1
        map[id] = idx[target_id];
    }
}


at::Tensor HashMap(const at::Tensor in_coords, const int k_size, at::Tensor in_map, at::Tensor kernel_nnz){

    int nnz = in_coords.size(0);
    int table_size = 2 * pow(2, ceil(log2((double)nnz)));
    int k_vol = k_size * k_size * k_size;

    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *in_map_ptr = in_map.data_ptr<int>();

    at::Tensor index = - torch::ones({table_size}, at::device(in_coords.device()).dtype(at::ScalarType::Int));
    at::Tensor value = torch::zeros({table_size}, at::device(in_coords.device()).dtype(at::ScalarType::Long));

    int *index_ptr = index.data_ptr<int>();
    uint64_t *value_ptr = (uint64_t *)(value.data_ptr<int64_t>());

    insertHash<<<dim3((nnz + BLOCK_SIZE  - 1) / BLOCK_SIZE, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(
            nnz, 
            table_size, 
            in_coords_ptr, 
            index_ptr
        );
    
        insertVal<<<dim3((table_size + BLOCK_SIZE  - 1) / BLOCK_SIZE, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(
            nnz, 
            table_size, 
            in_coords_ptr, 
            index_ptr, 
            value_ptr
        );
    
        queryHash<<<dim3((nnz * k_vol + BLOCK_SIZE  - 1) / BLOCK_SIZE, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(
            nnz, 
            table_size, 
            in_coords_ptr,
            k_size, 
            k_vol, 
            value_ptr, 
            index_ptr, 
            in_map_ptr
        );
    
    for (int k = 0; k < k_vol; ++k){

        at::Tensor kernel_map;
        kernel_map = torch::from_blob(&in_map_ptr[k * nnz], {nnz}, at::device(in_map.device()).dtype(at::ScalarType::Int));
        at::Tensor nnz_idx = torch::nonzero(kernel_map + torch::ones_like(kernel_map));  // torch::nonzero returns long tensor
        kernel_nnz[k] = nnz_idx.size(0);
    
    }

    at::Tensor whole_idx = torch::nonzero(in_map + torch::ones_like(in_map));

    return whole_idx;
}  