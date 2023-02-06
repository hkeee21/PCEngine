#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda/pipeline>

#define DIV_UP(x, y) (x + y - 1) / y
#define _FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define _FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
#define _HALF2(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])

/*******************************************************************
device functions
*/
__device__ __forceinline__ int binary_search(
                            const int *S_csrRowPtr, const int eid, 
                            const int start, const int end) {
    
    int lo = start, hi = end;
    if (lo == hi){
        return lo;
    }
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(S_csrRowPtr + mid) <= eid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (__ldg(S_csrRowPtr + hi) <= eid) {
        return hi;
    } else {
        return hi - 1;
    }
}

__device__ __forceinline__ float4 addFLOAT4(float4 a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;

    return a;
}


/*******************************************************************
gather for datatypes: float and half
*/
__global__ void gather_all_input_major_csr_float_4(
                    const int nnz, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x) << 2;
            if (cp >= c_in){break;}
            _FLOAT4(g_f[buf_pos * c_in + cp]) = 
                _FLOAT4(in_f[id * c_in + cp]);
        }
    }
}

__global__ void gather_all_input_major_csr_float_2(
                    const int nnz, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x) << 1;
            if (cp >= c_in){break;}
            _FLOAT2(g_f[buf_pos * c_in + cp]) = 
                _FLOAT2(in_f[id * c_in + cp]);
        }
    }
}


__global__ void gather_all_input_major_csr_float_2aligned4(
                    const int nnz, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
    const int c_mod = (c_in + 3) / 4 * 4;
    float pad[2] = {0.0f};
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x) << 1;
            if (cp >= c_mod){break;}
            _FLOAT2(g_f[buf_pos * c_mod + cp]) = (cp < c_in) ? 
                _FLOAT2(in_f[id * c_in + cp]) : _FLOAT2(pad[0]);
        }
    }
}


__global__ void gather_all_input_major_csr_half_8(
                    const int nnz, 
                    const int c_in, 
                    half *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    half *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x) << 3;
            if (cp >= c_in){break;}
            _FLOAT4(g_f[buf_pos * c_in + cp]) = 
                _FLOAT4(in_f[id * c_in + cp]);
        }
    }
}

__global__ void gather_all_input_major_csr_half_4(
                    const int nnz, 
                    const int c_in, 
                    half *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    half *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x) << 2;
            if (cp >= c_in){break;}
            _FLOAT2(g_f[buf_pos * c_in + cp]) = 
                _FLOAT2(in_f[id * c_in + cp]);
        }
    }
}

__global__ void gather_all_input_major_csr_half_2(
                    const int nnz, 
                    const int c_in, 
                    half *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    half *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x) << 1;
            if (cp >= c_in){break;}
            _HALF2(g_f[buf_pos * c_in + cp]) = 
                _HALF2(in_f[id * c_in + cp]);
        }
    }
}


__global__ void gather_all_input_major_csr_half_2aligned8(
                    const int nnz, 
                    const int c_in, 
                    half *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    half *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
    const int c_mod = (c_in + 7) / 8 * 8;
    half pad[2] = {__float2half(0.0f)};
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x) << 1;
            if (cp >= c_mod){break;}
            _HALF2(g_f[buf_pos * c_in + cp]) = (cp < c_in) ? 
                _HALF2(in_f[id * c_in + cp]) : _HALF2(pad[0]);
        }
    }
}


__global__ void gather_all_input_major_csr_half_1(
                    const int nnz, 
                    const int c_in, 
                    half *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    half *g_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += blockDim.x){
            // which input channel
            int cp = (c + threadIdx.x);
            if (cp >= c_in){break;}
            g_f[buf_pos * c_in + cp] = 
                in_f[id * c_in + cp];
        }
    }
}


/*******************************************************************
scatter for datatypes: float and half
*/
__global__ void scatter_all_output_major_csr_float(
                    const int nnz, 
                    const int c_out, 
                    float *s_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.y + threadIdx.y;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&ocsr[id]);
    const int m_end = __ldg(&ocsr[id + 1]);
    // working space
    float acc[4];
#pragma unroll
    for (int c = 0; ; c += blockDim.x){
        // which output channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
        // initialization
#pragma unroll
        for (int ofs = 0; ofs < 4; ofs++){
            acc[ofs] = 0.0f;
        }
#pragma unroll
        for (int k = m_start; k < m_end; k++){
            // which kernel offset
            int kinf = __ldg(&omap[k]);
            int kofs = kinf / 1186111;
            int buf_ofs = kinf % 1186111;
            int buf_start = __ldg(&kpos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            _FLOAT4(acc[0]) = addFLOAT4(
                _FLOAT4(acc[0]), 
                _FLOAT4(s_f[buf_pos * c_out + cp]));
        }
        _FLOAT4(out_f[id * c_out + cp]) = _FLOAT4(acc[0]);
    }
}


__global__ void scatter_all_output_major_csr_half(
                    const int nnz, 
                    const int c_out, 
                    half *s_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    half *out_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.y + threadIdx.y;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&ocsr[id]);
    const int m_end = __ldg(&ocsr[id + 1]);
    // working space
    half2 tmp[4];
    half2 acc[4];
#pragma unroll
    for (int c = 0; ; c += blockDim.x){
        // which output channel
        int cp = (c + threadIdx.x) << 3;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
        // initialization
#pragma unroll
        for (int ofs = 0; ofs < 4; ofs++){
            acc[ofs].x = __float2half(0.0f);
            acc[ofs].y = __float2half(0.0f);
        }
#pragma unroll
        for (int k = m_start; k < m_end; k++){
            // which kernel offset
            int kinf = __ldg(&omap[k]);
            int kofs = kinf / 1186111;
            int buf_ofs = kinf % 1186111;
            int buf_start = __ldg(&kpos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            // store 4 float type (8 half type) data into tmp
            _FLOAT4(tmp[0]) = _FLOAT4(s_f[buf_pos * c_out + cp]);
            // convert and accumulate temp into acc
#pragma unroll
            for(int t = 0; t < 4; t++){
                acc[t] = __hadd2(acc[t], tmp[t]);
            }      
        }
        _FLOAT4(out_f[id * c_out + cp]) = _FLOAT4(acc[0]);
    }
}



/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp32_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Load the matrices from device memory
  // to shared memory; each thread loads
  // one element of each matrix

  // Kernel weight to Bs
  *((float4*)(&Bs[ty][ctx])) = ((ty) < c_in && cx < c_out) ? 
    *((float4*)(kw_ptr + c_out * (ty) + cx)) : 
    *((float4*)(&padding[0]));
    
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    int y_temp = y + n * BLOCK_SIZE;

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float4*)(&As[n][ty][ctx])) = ((ctx) < c_in && in_row > -1) ? 
      *((float4*)(&in_f[c_in * in_row + ctx])) : 
      *((float4*)(&padding[0]));
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k) {
      float Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] += Ast * Bs[k][ctx + c];
      }
    }
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, 
blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void gemm_float_fused_largeN_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][2] = {0.0f};
  float padding[2] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, blockDim.x = 8, blockDim.y = 16.
The unaligned version deals with c_in % 4 != 0.
*/
template <int BLOCK_SIZE>
__global__ void gemm_float_fused_unaligned_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // The thread deals with the x-th channel of the y-th output
  const int out_row = y < __ldg(&kpos[widx + 1]) ? omap[y] : -1;
  const int in_row = y < __ldg(&kpos[widx + 1]) ? imap[y] : -1;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[2] = {0.0f, 0.0f};
  float padding[2] = {0.0f, 0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    *((float2*)(&As[ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
      *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
      *((float2*)(&padding[0]));
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      float Ast = As[ty][k];
#pragma unroll
      for (int c = 0; c < 2; c++){
        Csub[c] += Ast * Bs[k][ctx + c];
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (out_row > -1 && cx < c_out){
#pragma unroll
    for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx + c], Csub[c]);
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][2] = {__float2half(0.0f)};
  half padding[2] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void naive_gemm_fp16_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  // y-th valid nnz.
  const int qy = BLOCK_SIZE * N_LOOP * by + ty;
  const int y = qy - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]); 

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][2] = {__float2half(0.0f)};
  half padding[2] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int qy_temp = qy + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      bool in_valid = y_temp < __ldg(&kpos[widx + 1]);

      *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_valid) ? 
        *((float*)(&in_f[c_in * qy_temp + s + ctx])) : 
        *((float*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int qy_temp = qy + n * BLOCK_SIZE;
    bool out_valid = y_temp < __ldg(&kpos[widx + 1]);
    if (out_valid && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * qy_temp + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void naive_gemm_fp32_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  // y-th valid nnz.
  const int qy = BLOCK_SIZE * N_LOOP * by + ty;
  const int y = qy - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]); 

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][2] = {0.0f};
  float padding[2] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int qy_temp = qy + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      bool in_valid = y_temp < __ldg(&kpos[widx + 1]);

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_valid) ? 
        *((float2*)(&in_f[c_in * qy_temp + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int qy_temp = qy + n * BLOCK_SIZE;
    bool out_valid = y_temp < __ldg(&kpos[widx + 1]);
    if (out_valid && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * qy_temp + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fetch_on_demand_gemm_fp16_4_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][4] = {__float2half(0.0f)};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Kernel weight to Bs
  *((float2*)(&Bs[ty][ctx])) = (ty < c_in && cx < c_out) ? 
    *((float2*)(kw_ptr + c_out * ty + cx)) : 
    *((float2*)(&padding[0]));
    
  int y_temp = y;
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float2*)(&As[n][ty][ctx])) = (ctx < c_in && in_row > -1) ? 
      *((float2*)(&in_f[c_in * in_row + ctx])) : 
      *((float2*)(&padding[0]));
      
    y_temp += BLOCK_SIZE;
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k){
      half Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
      }
    }

    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


using namespace nvcuda;
/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_tf32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = wmma::__float_to_tf32(a[n].x[t]);
        }
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_fp16_tc4(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_fp16_tc4_async(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Pipelined copy between gmem and shmem
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(float2)>(sizeof(float2));

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    // const half *kw2Bs_ptr = ((s + ty) < c_in && cx < c_out) ? 
    //   kw_ptr + c_out * (s + ty) + cx : &padding[0];
    pipe.producer_acquire();
    if ((s + ty) < c_in && cx < c_out){
      cuda::memcpy_async(&Bs[ty][ctx], kw_ptr + c_out * (s + ty) + cx, shape4, pipe);
    }
    else{
      cuda::memcpy_async(&Bs[ty][ctx], &padding[0], shape4, pipe);
    }
    // cuda::memcpy_async(&Bs[ty][ctx], kw2Bs_ptr, shape4, pipe);
    pipe.producer_commit();
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      // const half *inf2As_ptr = ((s + ctx) < c_in && in_row > -1) ? 
      //   &in_f[c_in * in_row + s + ctx] : &padding[0];
      pipe.producer_acquire();
      if ((s + ctx) < c_in && in_row > -1){
        cuda::memcpy_async(&As[n][ty][ctx], &in_f[c_in * in_row + s + ctx], shape4, pipe);
      }
      else{
        cuda::memcpy_async(&As[n][ty][ctx], &padding[0], shape4, pipe);
      }
      // cuda::memcpy_async(&As[n][ty][ctx], inf2As_ptr, shape4, pipe);
      pipe.producer_commit();
    }

    // Synchronize to make sure the matrices are loaded
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
      }  
#pragma unroll 
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    pipe.consumer_release();
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void fetch_on_demand_gemm_fp16_tc4_async_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Pipelined copy between gmem and shmem
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(float2)>(sizeof(float2));

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }

  __syncthreads();
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

  // Kernel weight to Bs
  // const half *kw2Bs_ptr = ((s + ty) < c_in && cx < c_out) ? 
  //   kw_ptr + c_out * (s + ty) + cx : &padding[0];
  pipe.producer_acquire();
  if (ty < c_in && cx < c_out){
    cuda::memcpy_async(&Bs[ty][ctx], kw_ptr + c_out * ty + cx, shape4, pipe);
  }
  else{
    cuda::memcpy_async(&Bs[ty][ctx], &padding[0], shape4, pipe);
  }
  // cuda::memcpy_async(&Bs[ty][ctx], kw2Bs_ptr, shape4, pipe);
  pipe.producer_commit();
    
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    int y_temp = y + n * BLOCK_SIZE;

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    // const half *inf2As_ptr = ((s + ctx) < c_in && in_row > -1) ? 
    //   &in_f[c_in * in_row + s + ctx] : &padding[0];
    pipe.producer_acquire();
    if (ctx < c_in && in_row > -1){
      cuda::memcpy_async(&As[n][ty][ctx], &in_f[c_in * in_row + ctx], shape4, pipe);
    }
    else{
      cuda::memcpy_async(&As[n][ty][ctx], &padding[0], shape4, pipe);
    }
    // cuda::memcpy_async(&As[n][ty][ctx], inf2As_ptr, shape4, pipe);
    pipe.producer_commit();
  }

  // Synchronize to make sure the matrices are loaded
  cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();

  // Multiply the two matrices together using Tensor Core
  // Load data from shmem to tensor core
  // Just load Bs once
#pragma unroll
  for (int k = 0; k < BLOCK_SIZE; k += K){
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
    wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
    for (int n = 0; n < N_LOOP / 2; n++){
      wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
    }  
#pragma unroll 
    for (int n = 0; n < N_LOOP / 2; n++){
      wmma::mma_sync(c[n], a[n], b, c[n]);
    }
  }
  // Synchronize to make sure that the preceding
  // computation is done before loading two new
  // sub-matrices of A and B in the next iteration
  pipe.consumer_release();
  __syncthreads();

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void gemm_TF32_fused_largeN_skew_async(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Pipelined copy between gmem and shmem
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    if ((s + ty) < c_in && cx < c_out){
      pipe.producer_acquire();
      cuda::memcpy_async(&Bs[ty][ctx], (kw_ptr + c_out * (s + ty) + cx), shape4, pipe);
      pipe.producer_commit();
    }
    else{
      pipe.producer_acquire();
      cuda::memcpy_async(&Bs[ty][ctx], &padding[0], shape4, pipe);
      pipe.producer_commit();
    }
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      if ((s + ctx) < c_in && in_row > -1){
        pipe.producer_acquire();
        cuda::memcpy_async(&As[n][ty][ctx], &in_f[c_in * in_row + s + ctx], shape4, pipe);
        pipe.producer_commit();
      }
      else{
        pipe.producer_acquire();
        cuda::memcpy_async(&As[n][ty][ctx], &padding[0], shape4, pipe);
        pipe.producer_commit();
      }
    }

    // Synchronize to make sure the matrices are loaded
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = wmma::__float_to_tf32(a[n].x[t]);
        }
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    pipe.consumer_release();
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
}

