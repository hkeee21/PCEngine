#define COLLISION_BOUND 20

extern "C"


inline __device__ int buffer_encoder(const int k_id, const int k_map_id){
    return (k_id * 1186111 + k_map_id);
}

inline __device__ uint64_t coord_hash(const int b, const int ix, const int iy, const int iz){
    // +1 to avoid val==0
    return ((uint64_t)b * 23783141 + (uint64_t)ix * 73856093 
        + (uint64_t)iy * 19349669 + (uint64_t)iz * 83492791 + 1);
}

inline __device__ uint64_t shift_hash(const int size, const uint64_t value){
    return ((value + 1) % ((uint64_t)size - 2));
}

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


__global__ void insertHash(
                const int nnz, 
                const int size, 
                const int *__restrict__ coord, 
                int *idx){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // exclude illegal id number
    if (id >= nnz){return;}
    uint64_t temp_val = coord_hash(coord[4 * id], coord[4 * id + 1], 
        coord[4 * id + 2], coord[4 * id + 3]);
    // temp_val is unique
    uint64_t table_id = temp_val % (uint64_t)size;
    // cuckoo hashing
    int old_idx = atomicExch(&idx[table_id], id);
    // handle collision
    while(old_idx > -1){
        table_id = (table_id + 97) % size;
        old_idx = atomicExch(&idx[table_id], old_idx);
    }  
}

__global__ void insertVal(
                const int nnz, 
                const int size, 
                const int *__restrict__ coord, 
                const int *idx, 
                uint64_t *val){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size){return;}
    int input_id = idx[id];
    if (input_id < nnz && input_id > -1){
        val[id] = coord_hash(coord[4 * input_id], coord[4 * input_id + 1], 
            coord[4 * input_id + 2], coord[4 * input_id + 3]);
    }
}


template<int _NNZS_PER_BLOCK, int _KOFS_THREADS>
__global__ void queryHash_wholemap(
    // input nnz
    const int innz, 
    // output nnz
    const int onnz, 
    // input coords hash table size
    const int size, 
    // output coords, (onnz, 3)
    const int *__restrict__ coord,
    // coded kernel size, f = 311x + 17y + z
    const int ks_code, 
    // kernel volume
    const int kv, 
    // tensor stride
    const int t_stride_x, 
    const int t_stride_y, 
    const int t_stride_z, 
    // hash table (value), (size, )
    const uint64_t *val, 
    // hash table (index), (size, )
    const int *idx, 
    // input-major mapping, (innz * (kv - 1))
    int *imap,
    // output-major mapping, (onnz * (kv - 1))                 
    int *omap,  
    // the counter of nnz for each each kernel offsets, (kv - 1)               
    int *knnz,
    // whether to compute center kernel offset separately
    bool separate_mid)             
{
    // a thread for a coord
    int output_id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.y;  
    // exclude illegal id number
    if (output_id >= onnz){return;}
    // shared mem to store the output coord
    // TODO: with batch idx we can use int4 to read
    __shared__ int shm[_NNZS_PER_BLOCK * 4];
    shm[threadIdx.y * 4] = coord[output_id * 4];
    shm[threadIdx.y * 4 + 1] = coord[output_id * 4 + 1];
    shm[threadIdx.y * 4 + 2] = coord[output_id * 4 + 2];
    shm[threadIdx.y * 4 + 3] = coord[output_id * 4 + 3];
    // decode kernel size
    int ksx = ks_code / 311; 
    int ksy = (ks_code - ksx * 311) / 17;
    int ksz = ks_code - ksx * 311 - ksy * 17;
    int mid_ks = (ksx - 1) / 2 * ksy * ksz + 
        (ksy - 1) / 2 * ksz + (ksz - 1) / 2;
#pragma unroll
    for (int k = 0; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.x;
        // 0 <= kp < kv
        if (kp >= kv){break;}
        // ignore w[0, 0, 0]
        if (separate_mid && kp == mid_ks){continue;}
        int kx = kp / (ksz * ksy) - (ksx - 1) / 2;
        int ky = (kp / ksz) % ksy - (ksy - 1) / 2;
        int kz = kp % ksz - (ksz - 1) / 2;
        // expand kernel offsets by tensor stride
        kx *= t_stride_x;
        ky *= t_stride_y;
        kz *= t_stride_z;
        // hash query
        uint64_t target_val = coord_hash(shm[threadIdx.y * 4], 
            shm[threadIdx.y * 4 + 1] + kx, shm[threadIdx.y * 4 + 2] + ky, 
            shm[threadIdx.y * 4 + 3] + kz);
        uint64_t target_id = target_val % (uint64_t)size;
        // find target or empty
        while (val[target_id] != target_val && idx[target_id] > -1){
            target_id = (target_id + 97) % size;
        }
        // set map = input id or -1
        int input_id = idx[target_id];
        if(input_id < 0 || input_id >= innz){continue;}
        // writing into the map
        int buffer_pos = atomicAdd(&knnz[kp], 1);
        int buffer_code = buffer_encoder(kp, buffer_pos);
        imap[input_id * kv + kp] = buffer_code;
        omap[output_id * kv + kp] = buffer_code;
    }
}


__global__ void mapping_counter(
                const int nnz, 
                const int kv,
                const int *map,
                int *nnz_neighbor
){
    // a thread for a nnz
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if(nid >= nnz){return;}
    int counter = 0;
    for (int k = 0; k < kv; k++){
        bool effective = map[nid * kv + k] >= 0;
        counter += effective ? 1 : 0;
    }
    nnz_neighbor[nid] = counter;
}


__global__ void mapping_decoder(
                const int sum_nnz, 
                const int *__restrict__ kernel_pos,  
                int *map){
    // a thread for a mapping
    int mid = blockIdx.x * blockDim.x + threadIdx.x;
    if (mid >= sum_nnz){return;}
    int kinf = map[mid];
    int kofs = kernel_decoder(kinf);
    int buf_ofs = kernel_map_decoder(kinf);
    int buf_start = __ldg(&kernel_pos[kofs]);
    map[mid] = buf_start + buf_ofs;
}


/* 
The amount of kernel offsets is a small number [1, 125], 
so no more optimization is needed. The hand-written 
kernel can effectively remove the overhead to call 
Thrust::exclusive_scan function.
*/
__global__ void exclusive_scan_for_kernel(
                const int kv, 
                const int *input, 
                int *output
){
    // a thread for a scan
    const int id = threadIdx.x + 1;
    if (id >= kv){return;}
    float acc = 0.0f;
#pragma unroll 
    for (int i = 0; i < id; i++){  
        acc += input[i];
    }
    output[id] = acc;
}


/*
Make sure the coordinates are decoded by batch-first order.
*/
template <int _BOUND>
__global__ void coordsDownsample(
                // amount of non-zeros in input 
                const int innz,
                // stride of each dimension
                const int stride_x,
                const int stride_y, 
                const int stride_z, 
                // input coordinates, (innz, 4)
                const int *__restrict__ icoords, 
                // coded downsampled output coords, (innz, 1)
                uint64_t *ocoords_code
){
    const int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= innz) {return;}
    uint64_t code;
    // manually accumulation
    // code = b
    code = icoords[nid * 4];
    // code = b * s + x
    code = code * _BOUND + (icoords[nid * 4 + 1] / stride_x * stride_x);
    // code = (b * s + x) * s + y
    code = code * _BOUND + (icoords[nid * 4 + 2] / stride_y * stride_y);
    // code = ((b * s + x) * s + y) * s + z
    code = code * _BOUND + (icoords[nid * 4 + 3] / stride_z * stride_z);
    ocoords_code[nid] = code;
}


/*
The weights used for linear coding limit the coordinates to 
the range of [0, _BOUND).
*/
template <int _BOUND>
__global__ void coordsGenerator(
                // amount of non-zeros in output 
                const int onnz, 
                // coded downsampled output coords, (onnz, 1)
                const uint64_t *__restrict__ ocoords_code, 
                // decoded output coordinates, (onnz, 4)
                int *ocoords
){
    const int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= onnz) {return;}
    // TODO: coalesced memory access
    uint64_t code = ocoords_code[nid];
    // code = ((b * s + x) * s + y) * s + z
    ocoords[nid * 4 + 3] = code % _BOUND;
    code /= _BOUND;
    // code = (b * s + x) * s + y
    ocoords[nid * 4 + 2] = code % _BOUND;
    code /= _BOUND;
    // code = b * s + x
    ocoords[nid * 4 + 1] = code % _BOUND;
    code /= _BOUND;
    // code = b
    ocoords[nid * 4] = code;
}


