#include <cuda.h>
#include <cuda_runtime.h>

#define _FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

inline __device__ int kernel_decoder(int code){
    return (code / 1186111);
}

inline __device__ int kernel_map_decoder(int code){
    return (code % 1186111);
}

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


__global__ void gather_all_input_major_csr_template(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    const float *__restrict__ in_f, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * block_size_z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += block_size_y){
        int kp = k + threadIdx.y;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        int kinf = __ldg(&imap[kp]);
        // which kernel offset
        int kofs = kernel_decoder(kinf);
        int buf_ofs = kernel_map_decoder(kinf);
        int buf_start = __ldg(&knnz_pos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += block_size_x){
            // which input channel
            int cp = c + threadIdx.x;
            if (cp >= c_in){break;}
            (g_f[buf_pos * c_in + cp]) = 
                __ldg(&in_f[id * c_in + cp]);
        }
    }
}


__global__ void gather_all_input_major_csr_template3(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * block_size_z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += block_size_y){
        int kp = k + threadIdx.y;
        if (kp >= m_end){break;}
        int kinf = __ldg(&imap[kp]);
        // which kernel offset
        int kofs = kernel_decoder(kinf);
        int buf_ofs = kernel_map_decoder(kinf);
        int buf_start = __ldg(&knnz_pos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += block_size_x){
            // which input channel
            int cp = (c + threadIdx.x) << 2;
            if (cp >= c_in){break;}
            _FLOAT4(g_f[buf_pos * c_in + cp]) = _FLOAT4(in_f[id * c_in + cp]);
        }
    }
}


__global__ void gather_all_input_major_csr_template4(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * block_size_z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
    __shared__ float4 shm[block_size_x * block_size_z + 1];
#pragma unroll
    for (int c = 0; ; c += block_size_x){
        // which input channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_in){break;}
        shm[threadIdx.z * block_size_x + threadIdx.x] = 
            _FLOAT4(in_f[id * c_in + cp]);
#pragma unroll
        for (int k = m_start; ; k += block_size_y){
            int kp = k + threadIdx.y;
            if (kp >= m_end){break;}
            int kinf = __ldg(&imap[kp]);
            // which kernel offset
            int kofs = kernel_decoder(kinf);
            int buf_ofs = kernel_map_decoder(kinf);
            int buf_start = __ldg(&knnz_pos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            _FLOAT4(g_f[buf_pos * c_in + cp])
                = shm[threadIdx.z * block_size_x + threadIdx.x];
        }
    }
}


__global__ void gather_all_input_major_csr_template5(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * block_size_z + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int c = 0; ; c += block_size_x){
        // which input channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_in){break;}
#pragma unroll
        for (int k = m_start; ; k += block_size_y){
            int kp = k + threadIdx.y;
            if (kp >= m_end){break;}
            int kinf = __ldg(&imap[kp]);
            // which kernel offset
            int kofs = kernel_decoder(kinf);
            int buf_ofs = kernel_map_decoder(kinf);
            int buf_start = __ldg(&knnz_pos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            _FLOAT4(g_f[buf_pos * c_in + cp]) = _FLOAT4(in_f[id * c_in + cp]);
        }
    }
}