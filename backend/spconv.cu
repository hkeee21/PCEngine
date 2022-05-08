#include "spconv.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>

using namespace std;
#define XOFFSET 0
#define YOFFSET 1
#define ZOFFSET 2
#define BLOCK_SIZE 32

extern "C"

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)


__device__ bool search_axis(const int *coord, int *range, const int value, const int ao)
{    
    int start = range[0];
    int end = range[1];
    int mid = (start + end) / 2;
    while(coord[mid * 3 + ao] != value){
        //printf("start:%d, end:%d, mid:%d\n", start, end, mid);
        if (mid == start){
            if (coord[end * 3 + ao] == value){
                start = start + 1; range[0] = start; range[1] = end; return 1;}
            else if (coord[start * 3 + ao] == value){
                end = end - 1; range[0] = start; range[1] = end; return 1;}
            return 0;
        }
        if (coord[mid * 3 + ao] < value){
            start = mid;
            mid = (int)((start + end) / 2);
        }
        else{
            end = mid;
            mid = (int)((start + end) / 2);
        }
    }
    // TODO: The case with one target saves the following computation  
    int start_temp = start;
    int end_temp = end;
    while(coord[end * 3 + ao] > value){
        end = (int)((end + mid) / 2);
    }
    while(coord[end * 3 + ao] == value && end <= end_temp){
        end = end + 1;
    }
    end = end - 1;

    while(coord[start * 3 + ao] < value){
        start = ceil(float((start + mid)) / 2);
    }
    while(coord[start * 3 + ao] == value && start >= start_temp){
        start = start - 1;
    }
    start = start + 1;
    range[0] = start;
    range[1] = end;
    return 1;
}


__global__ void center_map(const int nnz, int *map)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord
    while(id < nnz)
    {
        map[2 * id] = id;
        map[2 * id + 1] = id;
        id += blockDim.x * gridDim.x;
    }

}


__global__ void search(const int nnz, const int *__restrict__ coord, 
                    const int ks, const int kx, const int ky, const int kz, int *map)
{   
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord
    
    while(id < nnz)
    {
        int Ix = coord[id * 3] + kx;
        int Iy = coord[id * 3 + 1] + ky;
        int Iz = coord[id * 3 + 2] + kz;

        /*if (kz > - ks / 2 && map[id] > -1){
            // use the existing mapping
            map[id] = coord[3 * (map[id] + 1) + 2] == Iz ? map[id] + 1 : -1;
            id += blockDim.x * gridDim.x;
            continue;
        }*/

        // initialize the mapping ...
        map[2 * id] = -1;
        map[2 * id + 1] = -1;

        if (Ix >= 0 && Iy >= 0 && Iz >= 0){
        // printf("Searching for coord (%d,%d,%d)\n", Ix, Iy, Iz);
        // search the corresponding input coord
        int range[2];
        range[0] = 0;
        range[1] = nnz - 1;

        while(range[0] < range[1]){
            // search for x
            if (search_axis(coord, range, Ix, XOFFSET) == 0){
                break;
            }
        
            // search for y
            if (search_axis(coord, range, Iy, YOFFSET) == 0){
                break;
            }
        
            // search for z
            if (search_axis(coord, range, Iz, ZOFFSET) == 1){
                if (range[0] == range[1]){
                    // printf("Input Z[%d] found. Final Found.\n", range[0]);
                    map[2 * id] = range[0];
                    break;
                }
                else{
                    // TODO: Methods settling for coord repetition
                    map[2 * id] = range[0];
                    // printf("Same coord appears twice, only one is used.\n");
                    break;
                }
            }
            else {
                break;
            }
        }
        //printf("Map[%d] derived.\n", id);
        }
        id += blockDim.x * gridDim.x;
    }
}


__global__ void query(const int nnz, int *map){

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    while(id < nnz)
    {
        int count = 0;
        int temp_map = map[2 * id];

        for(int i = 0; i < id; i++){
            if(map[2 * i] > -1){count += 1;}
        }

        __syncthreads();

        if (map[2 * id] > -1){

            map[2 * count + 1] = id;
            map[2 * count] = temp_map;
        }
        id += blockDim.x * gridDim.x;
    }
}


__global__ void gemm(const int nnz, const int c_in, const int c_out,
                const float *__restrict__ in_f, const float *__restrict__ kv, float *out_f,
                const int *map) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;

  // The thread deals with the x-th channel of the y-th output
  const int in_row = y < nnz ? map[2 * y] : -1;
  const int out_row = y < nnz ? map[2 * y + 1] : -1;

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
                            at::Tensor in_map, at::Tensor out_feats
                            ){
    
    // printf("[SubmanifoldSparseConv] - Starts.\n");

    int nnz = in_coords.size(0);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    int k_vol = k_size * k_size * k_size;

    size_t const blocknum = (nnz + BLOCK_SIZE  - 1) / BLOCK_SIZE;
    size_t const gridnum = nnz > out_channel ? (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE : (out_channel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *in_feats_ptr = in_feats.data_ptr<float>();
    float *weight_ptr = kernel.data_ptr<float>();
    float *out_feats_ptr = out_feats.data_ptr<float>();
    int *in_coords_ptr = in_coords.data_ptr<int>();
    int *in_map_ptr = in_map.data_ptr<int>();

    // Suppose an odd kernel size
    for (int i = 0; i < k_vol; i++){

        // calculate the kernel offset
        int k_offset_x = i / (k_size * k_size) - (k_size - 1) / 2;
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
        }

        query<<<dim3(blocknum, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(nnz, in_map_ptr);
        
        // GEMM
        gemm<<<dim3(gridnum, gridnum, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(
                nnz, 
                in_channel, out_channel,
                in_feats_ptr,
                &weight_ptr[i * in_channel * out_channel],
                out_feats_ptr,
                in_map_ptr);
    
    }

    // printf("[SubmanifoldSparseConv] - Ends.\n");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

}
