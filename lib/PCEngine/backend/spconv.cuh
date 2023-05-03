#include <cuda_runtime.h>
#include <cuda.h>

#define DIV_UP(x, y) (x + y - 1) / y
#define _FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define _FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
#define _HALF2(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])
#define CONVERT_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])

/*******************************************************************
device functions
*/
__device__ __forceinline__ int binary_search_find_nnz(
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
    const int c_mod = (c_in >> 2 + 1) << 2;
    float pad[2];
    pad[0] = 0.0f;
    pad[1] = 0.0f;
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
            if (cp >= c_in){
                _FLOAT2(g_f[buf_pos * c_mod + cp]) = 
                    _FLOAT2(pad[0]);
                break;
            }
            _FLOAT2(g_f[buf_pos * c_mod + cp]) = 
                _FLOAT2(in_f[id * c_in + cp]);
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
    float acc[4] = {0.0f};
#pragma unroll
    for (int c = 0; ; c += blockDim.x){
        // which output channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
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


/*************************************************
Ablation study on coded-CSR. 
*/
__global__ void gather_wo_csr(
                    const int nnz, 
                    const int c_in, 
                    const int k_vol, 
                    float *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ imap, 
                    float *g_f){
    
    // id-th nnz
    const int id = blockIdx.x * blockDim.z + threadIdx.z;  
    if (id >= nnz){return;}
    // const int m_start = __ldg(&icsr[id]);
    // const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = 0; ; k += blockDim.y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= k_vol){break;}
        // read the buffer position 
        int buf_ofs = __ldg(&imap[id * k_vol + kp]);
        if (buf_ofs < 0){continue;}
        int buf_start = __ldg(&kpos[kp]);
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


__global__ void scatter_wo_csr(
                    const int nnz, 
                    const int c_out, 
                    const int k_vol,
                    float *s_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * blockDim.y + threadIdx.y;  
    if (id >= nnz){return;}
    // const int m_start = __ldg(&ocsr[id]);
    // const int m_end = __ldg(&ocsr[id + 1]);
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
        for (int k = 0; k < k_vol; k++){
            // which kernel offset
            int buf_ofs = __ldg(&omap[id * k_vol + k]);
            if (buf_ofs < 0){continue;}
            int buf_start = __ldg(&kpos[k]);
            int buf_pos = buf_start + buf_ofs;
            _FLOAT4(acc[0]) = addFLOAT4(
                _FLOAT4(acc[0]), 
                _FLOAT4(s_f[buf_pos * c_out + cp]));
        }
        _FLOAT4(out_f[id * c_out + cp]) = _FLOAT4(acc[0]);
    }
}


__global__ void map2matrix(
                const int nnz,
                const int k_vol, 
                const int *__restrict__ csr,
                const int *__restrict__ map, 
                int *matrix){
  
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= nnz) {return;}
  int start = csr[id];
  int end = csr[id + 1];
#pragma unroll
  for (int k = start; k < end; k++){
    int kinf = __ldg(&map[k]);
    int kofs = kinf / 1186111;
    int buf_ofs = kinf % 1186111;
    matrix[id * k_vol + kofs] = buf_ofs;
  }
}

/*
gather & scatter kernels from torchsparse v2.0.0.
*/
// fused gather
template <typename scalar_t>
__global__ void gather_all_kernel_pad_sep_with_mask(
    const int n, const int c, const int kernel_volume, scalar_t *in_feat,
    scalar_t *out_feat, const int *cum_buffer_sizes,
    const int *input_mask, const int *output_mask, const bool transpose,
    const bool precompute_mid) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j;
  int offset = (sizeof(scalar_t) == 4) ? 2 : 3;
  i = index / (c >> offset);
  j = index % (c >> offset);
  if (i >= n) return;
  int4 tmps[1];
  CONVERT_INT4(tmps) = CONVERT_INT4(in_feat[i * c + (j << offset)]);
  if (transpose) {
    for (int k = 0; k < kernel_volume; k++) {
      // if(precompute_mid && k == kernel_volume / 2) continue;
      // int input_kmap_pos = input_mask[i * kernel_volume + k];
      //  another layout
      int input_kmap_pos = output_mask[k * n + i];
      if (input_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      // CONVERT_HALF2(out_feat[(cum_buffer_size + input_kmap_pos) * c + (j <<
      // 1)]) = CONVERT_HALF2(in_feat[i * c + (j << 1)]);
      CONVERT_INT4(
          out_feat[(cum_buffer_size + input_kmap_pos) * c + (j << offset)]) =
          tmps[0];
    }
  } else {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int input_kmap_pos = input_mask[i * kernel_volume + k];
      //  another layout
      int input_kmap_pos = input_mask[k * n + i];
      if (input_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      CONVERT_INT4(
          out_feat[(cum_buffer_size + input_kmap_pos) * c + (j << offset)]) =
          tmps[0];
    }
  }
}


__global__ void scatter_all_kernel_pad_sep_with_mask_float(
    const int n, const int c, const int kernel_volume, float *in_feat,
    float *out_feat, const int *cum_buffer_sizes,
    const int *input_mask, const int *output_mask, const bool transpose,
    const bool precompute_mid) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j;
  i = index / (c >> 2);
  j = index % (c >> 2);
  float tmp = 0.0f;
  if (i >= n) return;

  float tmps[4];
  int4 tmps_int4[1];
  for (int k = 0; k < 4; k++) tmps[k] = 0;
  if (i >= n) return;
  if (transpose) {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int output_kmap_pos = output_mask[i * kernel_volume + k];
      //  another layout
      int output_kmap_pos = input_mask[k * n + i];
      if (output_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      tmps_int4[0] = CONVERT_INT4(
          in_feat[(cum_buffer_size + output_kmap_pos) * c + (j << 2)]);
#pragma unroll
      for (int l = 0; l < 4; l++) {
        tmps[l] += *(reinterpret_cast<float *>(tmps_int4) + l);
      }
    }
  } else {
    for (int k = 0; k < kernel_volume; k++) {
      if (precompute_mid && k == kernel_volume / 2) continue;
      // int output_kmap_pos = output_mask[i * kernel_volume + k];
      //  another layout
      int output_kmap_pos = output_mask[k * n + i];
      if (output_kmap_pos < 0) continue;
      int cum_buffer_size = cum_buffer_sizes[k];
      tmps_int4[0] = CONVERT_INT4(
          in_feat[(cum_buffer_size + output_kmap_pos) * c + (j << 2)]);
#pragma unroll
      for (int l = 0; l < 4; l++) {
        tmps[l] += *(reinterpret_cast<float *>(tmps_int4) + l);
      }
    }
  }
  CONVERT_INT4(out_feat[i * c + (j << 2)]) = CONVERT_INT4(tmps);
  // float verify = *(reinterpret_cast<float*>(tmps) + 3);
  // printf("%f %f\n", verify, out_feat[i * c + (j << 2) + 3]);
}


/*************************************************
Fused GEMMs for D2, float and half
*/
template <int BLOCK_SIZE>
__global__ void horizontal_fused_gemm(
                const int nnz, 
                const int knum, 
                const int c_in,
                const int c_out,
                const int *__restrict__ kpos, 
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap,
                const int separate_mid) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  // dim z for the batch
  const int bz = blockIdx.z;
  const int kofs = (separate_mid < 0 || bz < separate_mid) ? bz : bz + 1;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = kpos[bz] + BLOCK_SIZE * by + ty;

  // The thread deals with the x-th channel of the y-th output
  const int out_row = y < (kpos[bz + 1]) ? omap[y] : -1;
  const int in_row = y < (kpos[bz + 1]) ? imap[y] : -1;

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
    As[ty][tx] = ((s + tx) < c_in && in_row > -1) ? in_f[c_in * in_row + s + tx] : 0;
    Bs[ty][tx] = ((s + ty) < c_in && x < c_out) ? kw[kofs * c_in * c_out + c_out * (s + ty) + x] : 0;

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
  if (out_row > -1 && x < c_out)
    atomicAdd(&out_f[c_out * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
}


template <int BLOCK_SIZE>
__global__ void block_fused_gemm_float(
                const int knum, 
                const int c_in,
                const int c_out,
                const int *__restrict__ kpos, 
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap
                ) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // which kernel offset
  int kidx = 0;
  int block_count = 0;
  int this_idx = 0;
  while(kidx < knum){
    this_idx = (__ldg(&kpos[kidx + 1]) - __ldg(&kpos[kidx]) 
        + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    if (block_count + this_idx > by) {break;}
    block_count += this_idx;
    kidx += 1;
  }

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = __ldg(&kpos[kidx]) + BLOCK_SIZE * (by - block_count) + ty;

  // The thread deals with the x-th channel of the y-th output
  const int out_row = y < (__ldg(&kpos[kidx + 1])) ? omap[y] : -1;
  const int in_row = y < (__ldg(&kpos[kidx + 1])) ? imap[y] : -1;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 8];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 8];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < c_in && in_row > -1) ? in_f[c_in * in_row + s + tx] : 0;
    Bs[ty][tx] = ((s + ty) < c_in && x < c_out) ? kw[kidx * c_in * c_out + c_out * (s + ty) + x] : 0;

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
  if (out_row > -1 && x < c_out)
    atomicAdd(&out_f[c_out * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
}


template <int BLOCK_SIZE>
__global__ void block_fused_gemm_half(
                const int knum, 
                const int c_in,
                const int c_out,
                const int *__restrict__ kpos, 
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap
                ) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  // which kernel offset
  int kidx = 0;
  int block_count = 0;
  int this_idx = 0;
  while(kidx < knum){
    this_idx = (__ldg(&kpos[kidx + 1]) - __ldg(&kpos[kidx]) 
        + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    if (block_count + this_idx > by) {break;}
    block_count += this_idx;
    kidx += 1;
  }

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = __ldg(&kpos[kidx]) + BLOCK_SIZE * (by - block_count) + ty;

  // The thread deals with the x-th channel of the y-th output
  const int out_row = y < (__ldg(&kpos[kidx + 1])) ? omap[y] : -1;
  const int in_row = y < (__ldg(&kpos[kidx + 1])) ? imap[y] : -1;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub = __float2half(0.0f);
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = ((s + tx) < c_in && in_row > -1) ? in_f[c_in * in_row + s + tx] : __float2half(0.0f);
    Bs[ty][tx] = ((s + ty) < c_in && x < c_out) ? kw[kidx * c_in * c_out + c_out * (s + ty) + x] : __float2half(0.0f);

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub = __hfma(As[ty][k], Bs[k][tx], Csub);
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  if (out_row > -1 && x < c_out)
    atomicAdd(&out_f[c_out * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fused_gemm_fp32_c4(
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
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, 
blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fused_gemm_fp32_c2(
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
BLOCK_SIZE = 16, N_LOOP = 2, SKEW = 8, 
blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fused_gemm_fp32_naive(
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
  const int ctx = tx;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

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
    *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float*)(&padding));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float*)(&padding));
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
        Csub[n] += Ast * Bs[k][ctx];
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
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fused_gemm_fp16_c4(
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
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
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
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void fused_gemm_fp16_c2(
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
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void batched_gemm_fp32_c4(
                const int *__restrict__ kpos, 
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
  const int bz = blockIdx.z;
  if (__ldg(&kpos[bz]) == __ldg(&kpos[bz + 1])) {return;}

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const float *kw_ptr = &kw[bz * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = __ldg(&kpos[bz]) + BLOCK_SIZE * N_LOOP * by + ty;

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
      int in_row = y_temp < __ldg(&kpos[bz + 1]) ? imap[y_temp] : -1;

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
    int out_row = y_temp < __ldg(&kpos[bz + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void separate_gemm_fp32_c4(
                const int *__restrict__ kpos, 
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
  int nnz = __ldg(&kpos[1]) - __ldg(&kpos[0]);
 
  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

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
      int in_row = y_temp < nnz ? imap[y_temp] : -1;

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
    int out_row = y_temp < nnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


