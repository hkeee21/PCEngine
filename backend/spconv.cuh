#define GEMM_SIZE 16
#define DIV_UP(x, y) (x + y - 1) / y
#define _FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])


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

__device__ __forceinline__ float4 addFLOAT4(float4 a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;

    return a;
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

            int kernel_idx = map_info / 1186111;
            int buffer_kidx = map_info % 1186111; 

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
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
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
__global__ void gather_all_input_major_csr_template(
                    const int nnz, 
                    const int kv, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.x;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
        // read the buffer position 
        int kinf = __ldg(&imap[kp]);
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&kpos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += _CHNS_THREADS){
            // which input channel
            int cp = (c + threadIdx.y) << 2;
            if (cp >= c_in){break;}
            _FLOAT4(g_f[buf_pos * c_in + cp]) = 
                _FLOAT4(in_f[id * c_in + cp]);
        }
    }
}


template <int _MPNS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void gather_all_input_major_csr_balance(
                    const int nnz, 
                    const int total_knnz,
                    const int kv, 
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // [m_start, m_end]-th mapping
    const int m_start = blockIdx.x * _MPNS_PER_BLOCK;  
    const int m_end = min(m_start + _MPNS_PER_BLOCK, total_knnz);
    // store the starting id of the block
    __shared__ int starting_id;
    starting_id = binary_search_find_nnz(icsr, m_start, 0, nnz);
    // nid[threadIdx.x] = binary_search_find_nnz(icsr, m_start, 0, nnz);
    // register to store the specific id of the thread
    // int nid = binary_search_find_nnz(icsr, m_start, 0, nnz);
    int nid = starting_id;
#pragma unroll
    for (int k = m_start; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.x;
        // make sure  m_start <= kp < m_end
        if (kp >= m_end){break;}
         // which nnz
        while (kp >= icsr[nid + 1]){
            nid += 1;}
        // read the buffer position
        int buf_pos = __ldg(&imap[kp]);
#pragma unroll
        for (int c = 0; ; c += _CHNS_THREADS){
            // which input channel
            int cp = (c + threadIdx.y) << 2;
            if (cp >= c_in){break;}
            _FLOAT4(g_f[buf_pos * c_in + cp]) = 
                _FLOAT4(in_f[nid * c_in + cp]);
        }    
    }
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void gather_all_input_major_csr_template2(
                    const int nnz, 
                    const int kv, 
                    const int c_in, 
                    const float *__restrict__ in_f, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz, threadIdx.z-th nnz within the block
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
    // store kofs for the nnz in the block
    extern __shared__ int shm[];
#pragma unroll
    for (int s = m_start; ; s += _KOFS_THREADS * _CHNS_THREADS){
        int sp = s + threadIdx.y * _CHNS_THREADS + threadIdx.x;
        // make sure m_start <= sp < m_end
        if (sp >= m_end){break;}
        shm[threadIdx.z * (kv - 1) + sp - m_start] = __ldg(&imap[sp]);
    }
    __syncthreads();
#pragma unroll
    for (int k = m_start; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.y;
        // make sure m_start <= kp < m_end
        if (kp >= m_end){break;}
        int buf_pos = shm[threadIdx.z * (kv - 1) + kp - m_start];
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
__global__ void gather_all_input_major_csr_template4(
                    const int nnz, 
                    const int kv,  
                    const int c_in, 
                    float *in_f, 
                    const int *__restrict__ icsr, 
                    const int *__restrict__ imap, 
                    float *g_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&icsr[id]);
    const int m_end = __ldg(&icsr[id + 1]);
    __shared__ float4 shm[_CHNS_THREADS * _NNZS_PER_BLOCK + 1];
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
        // which input channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_in){break;}
        shm[threadIdx.z * _CHNS_THREADS + threadIdx.x] = 
            _FLOAT4(in_f[id * c_in + cp]);
#pragma unroll
        for (int k = m_start; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            if (kp >= m_end){break;}
            int buf_pos = __ldg(&imap[kp]);
            _FLOAT4(g_f[buf_pos * c_in + cp])
                = shm[threadIdx.z * _CHNS_THREADS + threadIdx.x];
        }
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

            int kernel_idx = map_info / 1186111;
            int buffer_kidx = map_info % 1186111; 

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
        int kofs = kinf / 1186111;
        int buf_ofs = kinf % 1186111;
        int buf_start = __ldg(&knnz_pos[kofs]);
        int buf_pos = buf_start + buf_ofs;
#pragma unroll
        for (int c = 0; ; c += _CHNS_THREADS){
            // which output channel
            int cp = c + threadIdx.x;
            if (cp >= c_out){break;}
            atomicAdd(&out_f[id * c_out + cp], 
                s_f[buf_pos * c_out + cp]);
        }
    }
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_csr_template(
                    const int nnz, 
                    const int kv, 
                    const int c_out, 
                    const float *__restrict__ s_f, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&ocsr[id]);
    const int m_end = __ldg(&ocsr[id + 1]);
#pragma unroll
    for (int k = m_start; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.y;
        if (kp >= m_end){break;}
        // m_start <= kp < m_end
        int buf_pos = __ldg(&omap[kp]);
#pragma unroll
        for (int c = 0; ; c += _CHNS_THREADS){
            // which output channel
            int cp = c + threadIdx.x;
            if (cp >= c_out){break;}
            atomicAdd(&out_f[id * c_out + cp], 
                s_f[buf_pos * c_out + cp]);
        }
    }
}


template <int _MPNS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_csr_balance(
                    const int nnz, 
                    const int total_knnz,
                    const int kv, 
                    const int c_out, 
                    float *s_f, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // [m_start, m_end]-th mapping
    const int m_start = blockIdx.x * _MPNS_PER_BLOCK;  
    const int m_end = min(m_start + _MPNS_PER_BLOCK, total_knnz);
    // store the starting id
    __shared__ int starting_id;
    starting_id = binary_search_find_nnz(ocsr, m_start, 0, nnz);
    // working space
    float tmp[4];
    for (int c = 0; ; c += _CHNS_THREADS){
        // which input channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
        // id iterator
        int nid = starting_id;
        for (int k = m_start; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            // make sure  m_start <= kp < m_end
            if (kp >= m_end){break;}
            // which nnz
            while (kp >= ocsr[nid + 1]){
                nid += 1;}
            // read the buffer position
            int buf_pos = __ldg(&omap[kp]);
            _FLOAT4(tmp[0]) = _FLOAT4(s_f[buf_pos * c_out + cp]);
#pragma unroll
            for (int ofs = 0; ofs < 4; ofs++){
                atomicAdd(&(out_f[nid * c_out + cp + ofs]), 
                tmp[ofs]);
            }    
        }
    }

}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_csr_template2(
                    const int nnz, 
                    const int kv, 
                    const int c_out, 
                    const float *__restrict__ s_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&ocsr[id]);
    const int m_end = __ldg(&ocsr[id + 1]);
    // working space
    __shared__ float shm[_NNZS_PER_BLOCK * _CHNS_THREADS 
        * _KOFS_THREADS + 1];
    int working_offset = threadIdx.z * _CHNS_THREADS * _KOFS_THREADS + 
            threadIdx.x * _KOFS_THREADS + threadIdx.y;
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
        // which output channel
        int cp = c + threadIdx.x;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
        // initialization
        shm[working_offset] = 0;
#pragma unroll
        for (int k = m_start; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            // make sure  m_start <= kp < m_end
            if (kp >= m_end){break;}
            // which kernel offset
            int kinf = __ldg(&omap[kp]);
            int kofs = kinf / 1186111;
            int buf_ofs = kinf % 1186111;
            int buf_start = __ldg(&kpos[kofs]);
            int buf_pos = buf_start + buf_ofs;
            shm[working_offset] += 
                s_f[buf_pos * c_out + cp];
        }
        atomicAdd(&out_f[id * c_out + cp], 
            shm[working_offset]);
    }
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_csr_template3(
                    const int nnz, 
                    const int kv, 
                    const int c_out, 
                    float *s_f, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&ocsr[id]);
    const int m_end = __ldg(&ocsr[id + 1]);
    // working space
    __shared__ float shm[(_NNZS_PER_BLOCK * _CHNS_THREADS 
        * _KOFS_THREADS) * 4 + 1];
    int working_offset = (threadIdx.z * _CHNS_THREADS * _KOFS_THREADS + 
            threadIdx.x * _KOFS_THREADS + threadIdx.y) << 2;
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
        // which output channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
#pragma unroll
        for (int ofs = 0; ofs < 4; ofs++){
            shm[working_offset + ofs] = 0.0f;
        }
#pragma unroll
        for (int k = m_start; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            // make sure  m_start <= kp < m_end
            if (kp >= m_end){break;}
            // which kernel offset
            int buf_pos = __ldg(&omap[kp]);
            _FLOAT4(shm[working_offset]) = addFLOAT4(
                _FLOAT4(shm[working_offset]), 
                _FLOAT4(s_f[buf_pos * c_out + cp]));
        }
#pragma unroll
        for (int ofs = 0; ofs < 4; ofs++){
            atomicAdd(&(out_f[id * c_out + cp + ofs]), 
                shm[working_offset + ofs]);
        }
    }   
}


template <int _NNZS_PER_BLOCK, int _KOFS_THREADS, 
    int _CHNS_THREADS>
__global__ void scatter_all_output_major_csr_predecoding(
                    const int nnz, 
                    const int kv, 
                    const int c_out, 
                    float *s_f, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.z;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&ocsr[id]);
    const int m_end = __ldg(&ocsr[id + 1]);
    // working space
    __shared__ float shm[(_NNZS_PER_BLOCK * _CHNS_THREADS 
        * _KOFS_THREADS) * 4 + 1];
    int working_offset = (threadIdx.z * _CHNS_THREADS * _KOFS_THREADS + 
            threadIdx.x * _KOFS_THREADS + threadIdx.y) << 2;
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
        // which output channel
        int cp = (c + threadIdx.x) << 2;
        if (cp >= c_out){break;}
        // accumlated value for id-th nnz's cp-th channel
#pragma unroll
        for (int ofs = 0; ofs < 4; ofs++){
            shm[working_offset + ofs] = 0.0f;
        }
#pragma unroll
        for (int k = m_start; ; k += _KOFS_THREADS){
            int kp = k + threadIdx.y;
            // make sure m_start <= kp < m_end
            if (kp >= m_end){break;}
            // read the buffer position
            int buf_pos = __ldg(&omap[kp]);
            _FLOAT4(shm[working_offset]) = addFLOAT4(
                _FLOAT4(shm[working_offset]), 
                _FLOAT4(s_f[buf_pos * c_out + cp]));
        }
#pragma unroll
        for (int ofs = 0; ofs < 4; ofs++){
            atomicAdd(&(out_f[id * c_out + cp + ofs]), 
                shm[working_offset + ofs]);
        }
    }   
}


template <int _NNZS_PER_BLOCK, int _CHNS_THREADS>
__global__ void scatter_all_output_major_csr_t4k1(
                    const int nnz, 
                    const int kv, 
                    const int c_out, 
                    float *s_f, 
                    const int *__restrict__ kpos, 
                    const int *__restrict__ ocsr, 
                    const int *__restrict__ omap, 
                    float *out_f){
    // id-th nnz
    const int id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.y;  
    if (id >= nnz){return;}
    const int m_start = __ldg(&ocsr[id]);
    const int m_end = __ldg(&ocsr[id + 1]);
    // working space
    float acc[4];
#pragma unroll
    for (int c = 0; ; c += _CHNS_THREADS){
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
#pragma unroll
        for (int ofs = 0; ofs < 4; ofs++){
            out_f[id * c_out + cp + ofs] += acc[ofs];
        }
    }   
}

