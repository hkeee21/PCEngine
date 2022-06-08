
#define GEMM_SIZE 16

extern "C"

__global__ void center_map(const int nnz, int *map)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord
    while(id < nnz)
    {
        map[id] = id;

        id += blockDim.x * gridDim.x;
    }

}


__global__ void gather_all(const int nnz, const int total_nnz, const int c_in, 
                    const float *__restrict__ in_f, const long *idx, const int *map, float *g_f){
    
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_in;
    const int cid = id % c_in;
    const int bnid = (blockIdx.x * blockDim.x % c_in + threadIdx.x) / c_in;

    if (nid < total_nnz){

        extern __shared__ int in_row[];

        if (cid == 0 || threadIdx.x == 0){
            in_row[bnid] = map[idx[nid]];   // in_row 
        }

        __syncthreads();

        if (in_row[bnid] > -1 && in_row[bnid] < nnz){
            g_f[nid * c_in + cid] = in_f[in_row[bnid] * c_in + cid];
        } 
    }
}


__global__ void scatter_all(const int nnz, const int total_nnz, const int c_out,
                    const float *__restrict__ s_f, const long *idx, const int *map, float *out_f){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_out;
    const int cid = id % c_out;
    const int bnid = (blockIdx.x * blockDim.x % c_out + threadIdx.x) / c_out;

    if (nid < total_nnz){

        extern __shared__ int out_row[];

        if (cid == 0 || threadIdx.x == 0){
            out_row[bnid] = idx[nid] % nnz;   // out_row
        }

        __syncthreads();

        atomicAdd(&out_f[out_row[bnid] * c_out + cid], s_f[nid * c_out + cid]);
    }
}


__global__ void gather(const int nnz, const int kernel_nnz, const int c_in, 
                    const float *__restrict__ in_f, const long *idx, const int *map, float *g_f){
    
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_in;
    const int cid = id % c_in;
    const int bnid = (blockIdx.x * blockDim.x % c_in + threadIdx.x) / c_in;

    if (nid < kernel_nnz){

        extern __shared__ int in_row[];

        if (cid == 0 || threadIdx.x == 0){
            const int out_row = idx[nid] % nnz;   // out_row
            in_row[bnid] = map[out_row];   // in_row 
        }

        __syncthreads();

        if (in_row[bnid] > -1 && in_row[bnid] < nnz){
            g_f[nid * c_in + cid] = in_f[in_row[bnid] * c_in + cid];
        } 
    }
}


__global__ void scatter(const int nnz, const int kernel_nnz, const int c_out,
                    const float *__restrict__ s_f, const long *idx, const int *map, float *out_f){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_out;
    const int cid = id % c_out;
    const int bnid = (blockIdx.x * blockDim.x % c_out + threadIdx.x) / c_out;

    if (nid < kernel_nnz){

        extern __shared__ int out_row[];

        if (cid == 0 || threadIdx.x == 0){
            out_row[bnid] = idx[nid] % nnz;   // out_row
        }

        __syncthreads();

        atomicAdd(&out_f[out_row[bnid] * c_out + cid], s_f[nid * c_out + cid]);
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
  const int x = GEMM_SIZE * bx + tx;
  const int y = GEMM_SIZE * by + ty;
  // const int y = BLOCK_SIZE * bx + tx;
  // const int x = BLOCK_SIZE * by + ty;

  // The thread deals with the x-th channel of the y-th output
  const int out_row = y < kernel_nnz ? nnz_idx[y] % nnz : -1;
  const int in_row = y < kernel_nnz ? map[out_row] : -1;

  if(in_row > -1 && out_row > -1){
  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += GEMM_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[GEMM_SIZE][GEMM_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[GEMM_SIZE][GEMM_SIZE];

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
    for (int k = 0; k < GEMM_SIZE; ++k) {
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
