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

#define COLLISION_BOUND 20

extern "C"

inline __device__ int kernel_offset(const int kerIdx, const int coordIdx){
    int coord[26][3] = {
        {-1, -1, -1},  // 0
        { 1,  1,  1},  // 26
        {-1, -1,  0},  // 1
        { 1,  1,  0},  // 25
        {-1, -1,  1},  // 2
        { 1,  1, -1},  // 24
        {-1,  0, -1},  // 3
        { 1,  0,  1},  // 23
        {-1,  0,  0},  // 4
        { 1,  0,  0},  // 22
        {-1,  0,  1},  // 5
        { 1,  0, -1},  // 21
        {-1,  1, -1},  // 6
        { 1, -1,  1},  // 20
        {-1,  1,  0},  // 7
        { 1, -1,  0},  // 19
        {-1,  1,  1},  // 8
        { 1, -1, -1},  // 18
        { 0, -1, -1},  // 9 
        { 0,  1,  1},  // 17
        { 0, -1,  0},  // 10
        { 0,  1,  0},  // 16
        { 0, -1,  1},  // 11
        { 0,  1, -1},  // 15
        { 0,  0, -1},  // 12
        { 0,  0,  1}   // 14
    };
    return coord[kerIdx][coordIdx];
}

inline __device__ int buffer_encoder(const int k_id, const int k_map_id){
    return (k_id * 1186111 + k_map_id);
}

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


__global__ void queryHash(
                    const int nnz,
                    const int size, 
                    const int *__restrict__ coord,
                    const int ks, 
                    const int kv, 
                    const uint64_t *val, 
                    const int *idx,
                    int *map,                       // input-major map
                    int *knnz                        // nnz amounts of a certain kernel offset
                    ){

    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord

    int offset_pos_id = id / nnz;
    // 0, 1, ..., kv / 2 - 1, kv / 2 + 1, ..., kv - 1
    // int offset_id = offset_pos_id < kv / 2 ? offset_pos_id : offset_pos_id + 1;

    int nnz_id = id % nnz;

    // exclude illegal id number
    if (offset_pos_id < kv - 1)
    {
        // int kx = offset_id / (ks * ks) - (ks - 1) / 2;
        // int ky = (offset_id / ks) % ks - (ks - 1) / 2;
        // int kz = offset_id % ks - (ks - 1) / 2;
        int kx = kernel_offset(offset_pos_id, 0);
        int ky = kernel_offset(offset_pos_id, 1);
        int kz = kernel_offset(offset_pos_id, 2);

        int colli_num = 0; 

        int Ix = coord[nnz_id * 3] + kx;
        int Iy = coord[nnz_id * 3 + 1] + ky;
        int Iz = coord[nnz_id * 3 + 2] + kz;
        
        uint64_t target_val = coord_hash(Ix, Iy, Iz);

        uint64_t target_id = target_val % (uint64_t)size;

        // find target or empty
        while (val[target_id] != target_val && idx[target_id] > -1){
            colli_num += 1;
            if (colli_num == COLLISION_BOUND){return;}
            target_id = (target_id + 97) % size;
        }
        
        // set map = input id or -1
        int idx_to_write = idx[target_id];
        if(idx_to_write > -1){
            map[id] = idx_to_write;
            atomicAdd(&knnz[offset_pos_id], 1);
        }
    }    
}


__global__ void queryHash_wholemap(
                    const int nnz, 
                    const int size, 
                    const int *__restrict__ coord,
                    const int ks, 
                    const int kv, 
                    const uint64_t *val, 
                    const int *idx, 
                    int *imap,                 // input-major mapping, (nnz * (kv - 1))
                    int *omap,                 // output-major mapping, (nnz * (kv - 1))
                    int *knnz)                 // the counter of nnz for each each kernel offsets, (kv - 1)
{
    int output_id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord

    // the kernel offset id
    // int offset_pos_id = id / nnz;             
    // 0, 1, ..., kv / 2 - 1, kv / 2 + 1, ..., kv - 1
    // int offset_id = offset_pos_id < kv / 2 ? offset_pos_id : offset_pos_id + 1;

    // int output_id = id % nnz;

    // exclude illegal id number
    if (output_id < nnz)
    {
        int Ox = coord[output_id * 3];
        int Oy = coord[output_id * 3 + 1];
        int Oz = coord[output_id * 3 + 2];

        int Ocounter = 0;

        for (int k = 0; k < kv - 1; k++){
            
            int offset_id = k < kv / 2 ? k : k + 1;

            int kx = offset_id / (ks * ks) - (ks - 1) / 2;
            int ky = (offset_id / ks) % ks - (ks - 1) / 2;
            int kz = offset_id % ks - (ks - 1) / 2;
            // int kx = kernel_offset(offset_pos_id, 0);
            // int ky = kernel_offset(offset_pos_id, 1);
            // int kz = kernel_offset(offset_pos_id, 2);

            int colli_num = 0; 

            int Ix = Ox + kx;
            int Iy = Oy + ky;
            int Iz = Oz + kz;
        
            uint64_t target_val = coord_hash(Ix, Iy, Iz);

            uint64_t target_id = target_val % (uint64_t)size;

            // find target or empty
            while (val[target_id] != target_val && idx[target_id] > -1){
                colli_num += 1;
                if (colli_num == COLLISION_BOUND){continue;}
                target_id = (target_id + 97) % size;
            }
            // set map = input id or -1
            int input_id = idx[target_id];

            if(input_id < 0){continue;}

            int buffer_pos = atomicAdd(&knnz[k], 1);

            int buffer_code = buffer_encoder(k, buffer_pos);

            imap[input_id * (kv - 1) + k] = buffer_code;

            omap[output_id * (kv - 1) + Ocounter] = buffer_code; 

            Ocounter++;  
        }
    }
}


__global__ void queryHash_wholemap_stride1(
                    const int nnz, 
                    const int size, 
                    const int *__restrict__ coord,
                    const int ks, 
                    const int kv, 
                    const uint64_t *val, 
                    const int *idx, 
                    int *imap,                 // input-major mapping, (nnz * (kv - 1))
                    int *omap,                 // output-major mapping, (nnz * (kv - 1))
                    int *knnz)                 // the counter of nnz for each each kernel offsets, (kv - 1)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a nnz

    // the kernel offset id
    // int offset_pos_id = id / nnz;             
    // 0, 1, ..., kv / 2 - 1, kv / 2 + 1, ..., kv - 1
    // int offset_id = offset_pos_id < kv / 2 ? offset_pos_id : offset_pos_id + 1;

    // int output_id = id % nnz;

    // exclude illegal id number
    if (nid < nnz)
    {
        // attention: bank conflicts
        // extern __shared__ int counter[];

        int Nx = coord[nid * 3];
        int Ny = coord[nid * 3 + 1];
        int Nz = coord[nid * 3 + 2];

        for (int k = 0; k < (kv - 1) / 2; k++){
            
            int kx = k / (ks * ks) - (ks - 1) / 2;
            int ky = (k / ks) % ks - (ks - 1) / 2;
            int kz = k % ks - (ks - 1) / 2;
            // int kx = kernel_offset(k, 0);
            // int ky = kernel_offset(k, 1);
            // int kz = kernel_offset(k, 2);

            int colli_num = 0; 

            int Tx = Nx + kx;
            int Ty = Ny + ky;
            int Tz = Nz + kz;
        
            uint64_t target_val = coord_hash(Tx, Ty, Tz);
            
            uint64_t target_id = target_val % (uint64_t)size;

            // find target or empty
            while (val[target_id] != target_val && idx[target_id] > -1){
                colli_num += 1;
                if (colli_num == COLLISION_BOUND){continue;}
                target_id = (target_id + 97) % size;
            }
            // set map = id or -1
            int id_to_write = idx[target_id];

            if(id_to_write < 0){continue;}

            int buffer_o_pos = atomicAdd(&knnz[k], 1);
            int buffer_o_code = buffer_encoder(k, buffer_o_pos);

            int buffer_i_pos = atomicAdd(&knnz[kv - 2 - k], 1);
            int buffer_i_code = buffer_encoder(kv - 2 - k, buffer_i_pos);

            // id_to_write -- k -- nid
            omap[nid * (kv - 1) + k] = buffer_o_code;
            imap[id_to_write * (kv - 1) + k] = buffer_o_code;     

            // nid -- ( kv - 2 - k ) -- id_to_write
            omap[id_to_write * (kv - 1) + kv - 2 - k] = buffer_i_code;
            imap[nid * (kv - 1) + kv - 2 - k] = buffer_i_code;
        }
    }
}


__global__ void mapping_subtraction(
                const int nnz, 
                const int kv,
                int *map
){
    int nid = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a nnz

    if(nid < nnz){

        int counter = 0;

        for (int k = 0; k < kv - 1; k++){
            
            int effective_code = map[nid * (kv - 1) + k];

            if (effective_code < 0){continue;}

            map[nid * (kv - 1) + counter] = effective_code;

            if (k > counter){
                map[nid * (kv - 1) + k] = -1;
            }

            counter++;
        }
    }
}


void HashMap(const at::Tensor in_coords, 
                const int k_size, 
                at::Tensor imap,
                at::Tensor omap,  
                at::Tensor kernel_nnz){

    int nnz = in_coords.size(0);
    int table_size = 2 * pow(2, ceil(log2((double)nnz)));
    int k_vol = k_size * k_size * k_size;

    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *imap_ptr = imap.data_ptr<int>();
    int *omap_ptr = omap.data_ptr<int>();
    int *kernel_nnz_ptr = kernel_nnz.data_ptr<int>();

    at::Tensor index = - torch::ones({table_size}, at::device(in_coords.device()).dtype(at::ScalarType::Int));
    at::Tensor value = torch::zeros({table_size}, at::device(in_coords.device()).dtype(at::ScalarType::Long));

    int *index_ptr = index.data_ptr<int>();
    uint64_t *value_ptr = (uint64_t *)(value.data_ptr<int64_t>());

    insertHash<<<dim3((nnz + 15) / 16, 1, 1), dim3(16, 1, 1)>>>(
        nnz, 
        table_size, 
        in_coords_ptr, 
        index_ptr
    );
    
    insertVal<<<dim3((table_size + 15) / 16, 1, 1), dim3(16, 1, 1)>>>(
        nnz, 
        table_size, 
        in_coords_ptr, 
        index_ptr, 
        value_ptr
    );

    queryHash_wholemap<<<dim3((nnz + 15) / 16, 1, 1), dim3(16, 1, 1)>>>(
        nnz, 
        table_size, 
        in_coords_ptr,
        k_size, 
        k_vol, 
        value_ptr, 
        index_ptr, 
        imap_ptr,
        omap_ptr, 
        kernel_nnz_ptr
    );

    mapping_subtraction<<<dim3((nnz + 15) / 16, 1, 1), dim3(16, 1, 1)>>>(
        nnz,
        k_vol,
        imap_ptr
    );
    /*
    for (int k = 0; k < k_vol; ++k){

        at::Tensor kernel_map;
        kernel_map = torch::from_blob(&in_map_ptr[k * nnz], {nnz}, at::device(in_map.device()).dtype(at::ScalarType::Int));
        kernel_nnz[k] = torch::count_nonzero(kernel_map + torch::ones_like(kernel_map));
        // at::Tensor nnz_idx = torch::nonzero(kernel_map + torch::ones_like(kernel_map));
        // kernel_nnz[k] = nnz_idx.size(0);
    
    }*/

    // at::Tensor whole_idx = torch::nonzero(in_map + torch::ones_like(in_map));   // torch::nonzero returns long tensor

    // return whole_idx;
}  