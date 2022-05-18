#include "spconv.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <torch/extension.h>

using namespace std;
#define XOFFSET 0
#define YOFFSET 1
#define ZOFFSET 2
#define BLOCK_SIZE 32
#define COLLISION_BOUND 20

extern "C"

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)


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


__global__ void center_map(const int nnz, int *map)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord
    while(id < nnz)
    {
        map[id] = id;

        id += blockDim.x * gridDim.x;
    }

}


__global__ void gemm(const int nnz, const int kernel_nnz, const int c_in, const int c_out,
                const float *__restrict__ in_f, const float *__restrict__ kv, float *out_f,
                const long *nnz_idx, const int *map) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;
  // const int y = BLOCK_SIZE * bx + tx;
  // const int x = BLOCK_SIZE * by + ty;

  // The thread deals with the x-th channel of the y-th output
  const int out_row = y < kernel_nnz ? nnz_idx[y] : -1;
  const int in_row = y < kernel_nnz ? map[out_row] : -1;

  if(in_row > -1 && out_row > -1){
  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < c_in && in_row < nnz) ? in_f[c_in * in_row + s + tx] : 0;
    Bs[ty][tx] = ((s + ty) < c_in && x < c_out) ? kv[c_out * (s + ty) + x] : 0;

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (out_row < nnz && x < c_out)
    atomicAdd(&out_f[c_out * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
  }
}


void ConvolutionForward(at::Tensor in_coords, at::Tensor in_feats, 
                            at::Tensor kernel, const int k_size, 
                            at::Tensor in_map, at::Tensor out_feats,
                            const bool remap
                            ){
    
    // printf("[SubmanifoldSparseConv] - Starts.\n");

    int nnz = in_coords.size(0);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    int k_vol = k_size * k_size * k_size;

    int table_size = 2 * pow(2, ceil(log2((double)nnz)));

    float *in_feats_ptr = in_feats.data_ptr<float>();
    float *weight_ptr = kernel.data_ptr<float>();
    float *out_feats_ptr = out_feats.data_ptr<float>();
    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *in_map_ptr = in_map.data_ptr<int>();
    
    if(remap){

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
    }

    // Suppose an odd kernel size
    for (int i = 0; i < k_vol; i++){

        // calculate the kernel offset
        /*int k_offset_x = i / (k_size * k_size) - (k_size - 1) / 2;
        int k_offset_y = (i / k_size) % k_size - (k_size - 1) / 2;
        int k_offset_z = i % k_size - (k_size - 1) / 2;

        // search the nnz involved and record the mapping
        // kernel offset (0, 0, 0) need no mapping calculation

        if (k_offset_x == 0 && k_offset_y == 0 && k_offset_z == 0){
            center_map<<<dim3(blocknum, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(nnz, in_map_ptr);
        }
        else{
            search<<<dim3(blocknum, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(
                nnz, 
                in_coords_ptr, 
                k_size, 
                k_offset_x, k_offset_y, k_offset_z, 
                in_map_ptr);
            
            queryHash<<<dim3(blocknum, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(
                nnz, 
                table_size, 
                in_coords_ptr,
                k_size, 
                k_offset_x, 
                k_offset_y, 
                k_offset_z, 
                value_ptr, 
                index_ptr, 
                in_map_ptr
                );
        }*/
        
        at::Tensor kernel_map;
        kernel_map = torch::from_blob(&in_map_ptr[i * nnz], {nnz}, at::device(in_map.device()).dtype(at::ScalarType::Int));
        at::Tensor nnz_idx = torch::nonzero(kernel_map + torch::ones_like(kernel_map));  // torch::nonzero returns long tensor
        int kernel_nnz = nnz_idx.size(0);

        if (kernel_nnz == 0){continue;}

        // size_t const gridnum_x = (out_channel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // size_t const gridnum_y = (kernel_nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t const gridnum_x = (out_channel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t const gridnum_y = (kernel_nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // GEMM
        gemm<<<dim3(gridnum_x, gridnum_y, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(
                nnz, 
                kernel_nnz, 
                in_channel, out_channel,
                in_feats_ptr,
                &weight_ptr[i * in_channel * out_channel],
                out_feats_ptr,
                nnz_idx.data_ptr<long>(), 
                &in_map_ptr[i * nnz]);
    
    }

    // printf("[SubmanifoldSparseConv] - Ends.\n");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

}
