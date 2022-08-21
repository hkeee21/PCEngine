#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define _FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

inline __device__ int kernel_decoder(int code){
    return (code / 1186111);
}

inline __device__ int kernel_map_decoder(int code){
    return (code % 1186111);
}


__global__ void gather(const int nnz, const int kernel_nnz, const int c_in, 
                    const float *__restrict__ in_f, const int *imap, float *g_f){
    
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_in;
    const int cid = id % c_in;
    const int bnid = (blockIdx.x * blockDim.x % c_in + threadIdx.x) / c_in;

    if (nid < kernel_nnz){

        extern __shared__ int in_row[];

        if (cid == 0 || threadIdx.x == 0){
            in_row[bnid] = imap[nid];   // in_row 
        }

        __syncthreads();

        if (in_row[bnid] > -1 && in_row[bnid] < nnz){
            g_f[nid * c_in + cid] = in_f[in_row[bnid] * c_in + cid];
        } 
    }
}


__global__ void gather_all_input_major(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *knnz_pos, 
                    const int c_in, 
                    const float *__restrict__ in_f, 
                    const int *imap, 
                    float *g_f){
    
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_in;
    const int cid = id % c_in;
    // const int bnid = (blockIdx.x * blockDim.x % c_in + threadIdx.x) / c_in;

    if (nid < nnz){ 

        // extern __shared__ int binfo[];

        // TODO: shared mem
        for (int k = 0; k < kv - 1; k++){
            
            int map_info = imap[nid * (kv - 1) + k];

            if (map_info < 0) {break;} 

            int kernel_idx = kernel_decoder(map_info);
            int buffer_kidx = kernel_map_decoder(map_info); 

            int buffer_idx = buffer_kidx + knnz_pos[kernel_idx];

            if (buffer_idx < total_knnz){
                g_f[buffer_idx * c_in + cid] = in_f[nid * c_in + cid]; 
            }
        }
    }
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void gather_all_input_major_template(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    const float *__restrict__ in_f, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
#pragma unroll
    for (int k = 0; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.y;
        if (kp >= kv - 1){break;}
        int kinf = __ldg(&imap[id * (kv - 1) + kp]);
        if (kinf < 0){break;}
        // which kernel offset
        int kofs = kernel_decoder(kinf);
        int buf_ofs = kernel_map_decoder(kinf);
        int buf_start = __ldg(&knnz_pos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += _CHNS_THREADS){
            // which input channel
            int cp = c + threadIdx.x;
            if (cp >= c_in){break;}
            g_f[buf_pos * c_in + cp] = __ldg(&in_f[id * c_in + cp]);
        }
    }
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void gather_all_input_major_template2(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    const float *__restrict__ in_f, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    // tz-th nnz within the block
    const int tz = threadIdx.z;
    if (id >= nnz){return;}
    // store kofs for the nnz in the block
    extern __shared__ int shm[];
#pragma unroll
    for (int s = 0; ; s += _KOFS_THREADS * _CHNS_THREADS){
        int sp = s + threadIdx.y * _CHNS_THREADS + threadIdx.x;
        if (sp >= kv - 1){break;}
        // initialization
        shm[tz * (kv - 1) + sp] = -1;
        int kinf = __ldg(&imap[id * (kv - 1) + sp]);
        if (kinf < 0){break;}
        int kofs = kernel_decoder(kinf);
        int buf_ofs = kernel_map_decoder(kinf);
        int buf_start = __ldg(&knnz_pos[kofs]);
        shm[tz * (kv - 1) + sp] = buf_start + buf_ofs;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.y;
        if (kp >= kv - 1){break;}
        int buf_pos = shm[tz * (kv - 1) + kp];
        if (buf_pos < 0){break;}
        for (int c = 0; ; c += _CHNS_THREADS){
            // which input channel
            int cp = c + threadIdx.x;
            if (cp >= c_in){break;}
            g_f[buf_pos * c_in + cp] = __ldg(&in_f[id * c_in + cp]);
        }
    }
}


/*
Template3 has a 1.06x speedup over Template.
*/
template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void gather_all_input_major_template3(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
#pragma unroll
    for (int k = 0; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.y;
        if (kp >= kv - 1){break;}
        int kinf = __ldg(&imap[id * (kv - 1) + kp]);
        if (kinf < 0){break;}
        // which kernel offset
        int kofs = kernel_decoder(kinf);
        int buf_ofs = kernel_map_decoder(kinf);
        int buf_start = __ldg(&knnz_pos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += _CHNS_THREADS){
            // which input channel
            int cp = (c + threadIdx.x) << 2;
            if (cp >= c_in){break;}
            _FLOAT4(g_f[buf_pos * c_in + cp]) = _FLOAT4(in_f[id * c_in + cp]);
        }
    }
}

/*
Template4 is quicker than Template with small channels. 
Redundant kernel offset calculations exist.
*/
template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void gather_all_input_major_template4(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_in, 
                    const float *__restrict__ in_f, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
        // which input channel
        int cp = c + threadIdx.x;
        if (cp >= c_in){break;}
        float to_write = __ldg(&in_f[id * c_in + cp]);
#pragma unroll
        for (int k = 0; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            if (kp >= kv - 1){break;}
            int kinf = __ldg(&imap[id * (kv - 1) + kp]);
            if (kinf < 0){break;}
            // which kernel offset
            int kofs = kernel_decoder(kinf);
            int buf_ofs = kernel_map_decoder(kinf);
            int buf_start = __ldg(&knnz_pos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            g_f[buf_pos * c_in + cp] = to_write;
        }
    }
}


__global__ void scatter(const int nnz, const int kernel_nnz, const int c_out,
                    const float *__restrict__ s_f, const int *omap, float *out_f){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_out;
    const int cid = id % c_out;
    const int bnid = (blockIdx.x * blockDim.x % c_out + threadIdx.x) / c_out;

    if (nid < kernel_nnz){

        extern __shared__ int out_row[];

        if (cid == 0 || threadIdx.x == 0){
            out_row[bnid] = omap[nid];   // out_row
        }

        __syncthreads();

        atomicAdd(&out_f[out_row[bnid] * c_out + cid], s_f[nid * c_out + cid]);
    }
}


__global__ void scatter_all_output_major(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *knnz_pos, 
                    const int c_out, 
                    const float *__restrict__ s_f, 
                    const int *omap, 
                    float *out_f){

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int nid = id / c_out;
    const int cid = id % c_out;
    // const int bnid = (blockIdx.x * blockDim.x % c_out + threadIdx.x) / c_out;

    if (nid < nnz){

        for (int k = 0; k < kv - 1; k++){

            int map_info = omap[nid * (kv - 1) + k];

            if (map_info < 0) {break;} 

            int kernel_idx = kernel_decoder(map_info);
            int buffer_kidx = kernel_map_decoder(map_info); 

            int buffer_idx = buffer_kidx + knnz_pos[kernel_idx];

            if (buffer_idx < total_knnz){
                atomicAdd(&out_f[nid * c_out + cid], s_f[buffer_idx * c_out + cid]);
            }

        }
        // atomicAdd(&out_f[out_row[2 * bnid + 1] * c_out + cid], s_f[out_row[2 * bnid] * c_out + cid]);
    }
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_template(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_out, 
                    const float *__restrict__ s_f, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
#pragma unroll
    for (int k = 0; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.y;
        if (kp >= kv - 1){break;}
        int kinf = __ldg(&omap[id * (kv - 1) + kp]);
        if (kinf < 0){break;}
        // which kernel offset
        int kofs = kernel_decoder(kinf);
        int buf_ofs = kernel_map_decoder(kinf);
        int buf_start = __ldg(&knnz_pos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += _CHNS_THREADS){
            // which output channel
            int cp = c + threadIdx.x;
            if (cp >= c_out){break;}
            atomicAdd(&out_f[id * c_out + cp], s_f[buf_pos * c_out + cp]);
        }
    }
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_template2(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_out, 
                    const float *__restrict__ s_f, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    // working space
    __shared__ float shm[_NNZS_PER_BLOCK * _CHNS_THREADS * _KOFS_THREADS + 1];
    int working_offset = threadIdx.z * _CHNS_THREADS * _KOFS_THREADS + 
            threadIdx.x * _KOFS_THREADS;
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
        // which output channel
        int cp = c + threadIdx.x;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
        float Acc = 0.0f;
#pragma unroll
        for (int k = 0; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            if (kp >= kv - 1){break;}
            int kinf = __ldg(&omap[id * (kv - 1) + kp]);
            if (kinf < 0){break;}
            // which kernel offset
            int kofs = kernel_decoder(kinf);
            int buf_ofs = kernel_map_decoder(kinf);
            int buf_start = __ldg(&knnz_pos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            Acc += s_f[buf_pos * c_out + cp];
        }
        //_KOFS_THREADS registers to shared memory
        shm[working_offset + threadIdx.y] = Acc;
        // make sure the shared mem is ready
        __syncthreads();
        // sequential reduction within the chunk to avoid sync
        if (threadIdx.y > 0){continue;}
        for (int ac = 1; ac < _KOFS_THREADS; ac++){
            shm[working_offset] += shm[working_offset + ac];
        }
        out_f[id * c_out + cp] += shm[working_offset];
    }
}


/*
Template3 is slower than Template.
*/
template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_template3(
                    const int nnz, 
                    const int kv, 
                    const int total_knnz, 
                    const int *__restrict__ knnz_pos, 
                    const int c_out, 
                    const float *__restrict__ s_f, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
        // which output channel
        int cp = c + threadIdx.x;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
        float Acc = 0.0f;
#pragma unroll
        for (int k = 0; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            if (kp >= kv - 1){break;}
            int kinf = __ldg(&omap[id * (kv - 1) + kp]);
            if (kinf < 0){break;}
            // which kernel offset
            int kofs = kernel_decoder(kinf);
            int buf_ofs = kernel_map_decoder(kinf);
            int buf_start = __ldg(&knnz_pos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            Acc += s_f[buf_pos * c_out + cp];
        }
        atomicAdd(&out_f[id * c_out + cp], Acc);
    }
}


