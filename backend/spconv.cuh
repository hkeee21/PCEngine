
#define DIV_UP(x, y) (x + y - 1) / y
#define _FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define _HALF2(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])

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
__global__ void gather_all_input_major_csr_float(
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


__global__ void gather_all_input_major_csr_half(
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
