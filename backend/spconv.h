#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

#define XOFFSET 0
#define YOFFSET 1
#define ZOFFSET 2
#define BLOCK_SIZE 32

extern "C"

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


__global__ void search(const int nnz, const int c_in, const int c_out,
                        const int *__restrict__ coord, const int kx, const int ky, const int kz, 
                        int *map)
{   
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord
    while(id < nnz)
    {
        int Ix = coord[id * 3] + kx;
        int Iy = coord[id * 3 + 1] + ky;
        int Iz = coord[id * 3 + 2] + kz;

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
                    map[id] = range[0];
                    //atomicAdd(&map[id], z_start + 1);
                    break;
                }
                else{
                    printf("Same coord appears twice!");
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
  const int in_row = y < nnz ? map[y] : -1;

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
    As[ty][tx] = ((s + tx) < c_in && y < nnz && in_row != -1) ? in_f[c_in * in_row + s + tx] : 0;
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
  if (y < nnz && x < c_out)
    atomicAdd(&out_f[c_out * y + x], Csub);
  // C[wB * out_row + x] += Csub;
}
    

/*void conv_fwd_cuda(at::Tensor in_coords, at::Tensor in_feats, 
                            at::Tensor kernel, const int k_size, 
                            at::Tensor out_feats
                            ) {
    
    printf("[SubmanifoldSparseConv] - Starting...\n");

    int nnz = in_coords.size(0);
    int in_channel = in_feats.size(1);
    if (in_feats.size(1) != kernel.size(1)) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }
    int out_channel = kernel.size(2);
    
    size_t const blocknum = (nnz + BLOCKSIZE  - 1) / BLOCKSIZE ;

    auto weight_addr = weights.data_ptr<float>();

    size_t const gridnum = nnz > out_channel ? (nnz + BLOCKSIZE - 1) / BLOCKSIZE : (out_channel + BLOCKSIZE - 1) / BLOCKSIZE;

    // Suppose an odd kernel size
    for (i = 0; i < k_size ^ 3; i++){

        // calculate the kernel offset
        int k_offset_x = i / (k_size * k_size) - (k_size - 1) / 2;
        int k_offset_y = (i / k_size) % k_size - (k_size - 1) / 2;
        int k_offset_z = i % k_size - (k_size - 1) / 2;

        // allocate memory for mapping: (idx_in, idx_out)
        int *in_map;
        cudaMalloc((void**)&in_map, nnz * sizeof(int));
        cudaMemset(in_map, 0, nnz * sizeof(int));

        // search the nnz involved and record the mapping
        search<<<dim3(blocknum, 1, 1), dim3(BLOCKSIZE, 1, 1)>>>(
                nnz, 
                in_channel, out_channel,
                in_coords.data_ptr<int>(), 
                k_offset_x, k_offset_y, k_offset_z, 
                in_map);
        
        // GEMM
        gemm<<<dim3(gridnum, gridnum, 1), dim3(BLOCKSIZE, BLOCKSIZE, 1)>>>(
                nnz, 
                in_channel, out_channel,
                in_feats.data_ptr<float>(),
                &weight_addr[i * in_channel * out_channel],
                out_feats.data_ptr<float>(),
                in_map);

        cudaFree(in_map);
    
    }

}*/



void Conv_CPU(const int nnz, const int c_in, const int c_out, 
                const int *in_c, const float *in_f, const float *kv, 
                const int ks, float *out)
{
    for (int i = 0; i < nnz ; i++){
        for (int j = 0; j < nnz ; j++){
            int off_x = in_c[3 * i] - in_c[3 * j];
            int off_y = in_c[3 * i + 1] - in_c[3 * j + 1];
            int off_z = in_c[3 * i + 2] - in_c[3 * j + 2];
            if (abs(off_x) <= ks / 2 && abs(off_y) <= ks / 2 && abs(off_z) <= ks / 2){
                int kid = (off_x + ks / 2) * ks * ks + (off_y + ks / 2) * ks + off_z + ks / 2;
                // Conv operation
                for (int co = 0; co < c_out; co++){
                    float tv = 0;
                    for (int c = 0; c < c_in; c++){
                        tv += kv[kid * c_out * c_in + c * c_out + co] * in_f[i * c_in + c];
                    }
                    out[j * c_out + co] += tv;
                }
            }
        }
    }
    
    printf("Computation on CPU Done.\n");  
}


bool search_axis_cpu(int *coord, int *range, int value, int ao)
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


void prinf_variable(int *coord, int *range, int value, int ao){
    printf("range start: %d, range end %d\n", range[0], range[1]);
    for (int i = range[0]; i <= range[1]; i++){
        printf("coord: %d\n", coord[3 * i + ao]);
    }
    printf("value: %d\n", value);
}


void search_CPU(int nnz, int c_in, int c_out, int *coord, int kx, int ky, int kz, int *map)
{   
    // printf("Map search starts ...\n");
    for(int id = 0; id < nnz; id++)
    {
        int Ix = coord[id * 3] + kx;
        int Iy = coord[id * 3 + 1] + ky;
        int Iz = coord[id * 3 + 2] + kz;

        if (Ix >= 0 && Iy >= 0 && Iz >= 0){
        // printf("Searching for coord (%d,%d,%d)\n", Ix, Iy, Iz);
        // search the corresponding input coord
        int range[2];
        range[0] = 0;
        range[1] = nnz - 1;

        // search for x
        while(range[0] < range[1]){
            if (search_axis_cpu(coord, range, Ix, XOFFSET) == 0){
                break;
            }
            // if (search_axis_cpu(coord, range, Ix, XOFFSET) == 1){
            //     printf("Input X(%d, %d) found.\n", range[0], range[1]);
            // }
            // prinf_variable(coord, range, Iz, ZOFFSET);

        
            // search for y
            if (search_axis_cpu(coord, range, Iy, YOFFSET) == 0){
                break;
            }
        
            // search for z
            if (search_axis_cpu(coord, range, Iz, ZOFFSET) == 1){
                if (range[0] == range[1]){
                    // printf("Input Z[%d] found. Final Found.\n", range[0]);
                    map[id] = range[0];
                    //atomicAdd(&map[id], z_start + 1);
                    break;
                }
                else{
                    printf("Same coord appears twice!");
                }
            }
            else {
                break;
            }
        }
        //printf("Map[%d] derived.\n", id);
        }
    }
}


void gemm_cpu(const int nnz, const int c_in, const int c_out, const float *in_f, const float *kv, const int *map, float *out){
    
    for (int j = 0 ; j < nnz ; j++){
        int row_id = map[j];
        // printf("%d - row id: %d", j, row_id);
        if (row_id >= 0 && row_id < nnz){
            for (int co = 0; co < c_out; co++){
                float temp_out = 0;
                for (int c = 0; c < c_in; c++){
                    temp_out += in_f[row_id * c_in + c] * kv[c * c_out + co];
                }
                out[j * c_out + co] += temp_out;
            }
        }
    }

}
