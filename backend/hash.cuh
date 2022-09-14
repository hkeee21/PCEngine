#define COLLISION_BOUND 20

extern "C"

inline __device__ int kernel_offset(const int kerIdx, const int coordIdx){
    int coord[26][3] = {
        {-1, -1, -1},  // 0
        { 1,  1,  1},  // 26
        {-1, -1,  0},  // 1
        { 1,  1,  0},  // 25
        {-1, -1,  1},  // 2
        { 1,  1, -1},  // 24
        {-1,  0, -1},  // 3
        { 1,  0,  1},  // 23
        {-1,  0,  0},  // 4
        { 1,  0,  0},  // 22
        {-1,  0,  1},  // 5
        { 1,  0, -1},  // 21
        {-1,  1, -1},  // 6
        { 1, -1,  1},  // 20
        {-1,  1,  0},  // 7
        { 1, -1,  0},  // 19
        {-1,  1,  1},  // 8
        { 1, -1, -1},  // 18
        { 0, -1, -1},  // 9 
        { 0,  1,  1},  // 17
        { 0, -1,  0},  // 10
        { 0,  1,  0},  // 16
        { 0, -1,  1},  // 11
        { 0,  1, -1},  // 15
        { 0,  0, -1},  // 12
        { 0,  0,  1}   // 14
    };
    return coord[kerIdx][coordIdx];
}

inline __device__ int buffer_encoder(const int k_id, const int k_map_id){
    return (k_id * 1186111 + k_map_id);
}

inline __device__ uint64_t coord_hash(const int ix, const int iy, const int iz){
    // +1 to avoid val==0
    return ((uint64_t)ix * 73856093 + (uint64_t)iy * 19349669 + (uint64_t)iz * 83492791 + 1);
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


__global__ void insertHash(
                const int nnz, 
                const int size, 
                const int *__restrict__ coord, 
                int *idx){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // exclude illegal id number
    if (id >= nnz){return;}
    uint64_t temp_val = coord_hash(coord[3 * id], 
        coord[3 * id + 1], coord[3 * id + 2]);
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
        val[id] = coord_hash(coord[3 *input_id], 
            coord[3 * input_id + 1], coord[3 * input_id + 2]);
    }
}


__global__ void queryHash(
                    const int nnz,
                    const int size, 
                    const int *__restrict__ coord,
                    const int ks, 
                    const int kv, 
                    const uint64_t *val, 
                    const int *idx,
                    int *map,                       // input-major map
                    int *knnz                        // nnz amounts of a certain kernel offset
                    ){

    int id = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a coord

    int offset_pos_id = id / nnz;
    // 0, 1, ..., kv / 2 - 1, kv / 2 + 1, ..., kv - 1
    // int offset_id = offset_pos_id < kv / 2 ? offset_pos_id : offset_pos_id + 1;

    int nnz_id = id % nnz;

    // exclude illegal id number
    if (offset_pos_id < kv - 1)
    {
        // int kx = offset_id / (ks * ks) - (ks - 1) / 2;
        // int ky = (offset_id / ks) % ks - (ks - 1) / 2;
        // int kz = offset_id % ks - (ks - 1) / 2;
        int kx = kernel_offset(offset_pos_id, 0);
        int ky = kernel_offset(offset_pos_id, 1);
        int kz = kernel_offset(offset_pos_id, 2);

        int colli_num = 0; 

        int Ix = coord[nnz_id * 3] + kx;
        int Iy = coord[nnz_id * 3 + 1] + ky;
        int Iz = coord[nnz_id * 3 + 2] + kz;
        
        uint64_t target_val = coord_hash(Ix, Iy, Iz);

        uint64_t target_id = target_val % (uint64_t)size;

        // find target or empty
        while (val[target_id] != target_val && idx[target_id] > -1){
            colli_num += 1;
            if (colli_num == COLLISION_BOUND){return;}
            target_id = (target_id + 97) % size;
        }
        
        // set map = input id or -1
        int idx_to_write = idx[target_id];
        if(idx_to_write > -1){
            map[id] = idx_to_write;
            atomicAdd(&knnz[offset_pos_id], 1);
        }
    }    
}

template<int _NNZS_PER_BLOCK, int _KOFS_THREADS>
__global__ void queryHash_wholemap(
                    const int nnz, 
                    const int size, 
                    const int *__restrict__ coord,
                    const int ks, 
                    const int kv, 
                    const uint64_t *val, 
                    const int *idx, 
                    int *imap,                 // input-major mapping, (nnz * (kv - 1))
                    int *omap,                 // output-major mapping, (nnz * (kv - 1))
                    int *knnz)                 // the counter of nnz for each each kernel offsets, (kv - 1)
{
    // a thread for a coord
    int output_id = blockIdx.x * _NNZS_PER_BLOCK + threadIdx.x;  
    // exclude illegal id number
    if (output_id >= nnz){return;}
    // shared mem to store the output coord
    __shared__ int shm[_NNZS_PER_BLOCK * 3];
    shm[threadIdx.x * 3] = coord[output_id * 3];
    shm[threadIdx.x * 3 + 1] = coord[output_id * 3 + 1];
    shm[threadIdx.x * 3 + 2] = coord[output_id * 3 + 2];
#pragma unroll
    for (int k = 0; ; k += _KOFS_THREADS){
        int kp = k + threadIdx.y;
        if (kp >= kv - 1){break;}
        // 0 <= kp < kv - 1
        int offset_id = kp < kv / 2 ? kp : kp + 1;
        int kx = offset_id / (ks * ks) - (ks - 1) / 2;
        int ky = (offset_id / ks) % ks - (ks - 1) / 2;
        int kz = offset_id % ks - (ks - 1) / 2;
        // hash query
        uint64_t target_val = coord_hash(shm[threadIdx.x * 3] + kx, 
            shm[threadIdx.x * 3 + 1] + ky, shm[threadIdx.x * 3 + 2] + kz);
        uint64_t target_id = target_val % (uint64_t)size;
        // find target or empty
        while (val[target_id] != target_val && idx[target_id] > -1){
            target_id = (target_id + 97) % size;
        }
        // set map = input id or -1
        int input_id = idx[target_id];
        if(input_id < 0){continue;}
        // writing
        int buffer_pos = atomicAdd(&knnz[kp], 1);
        int buffer_code = buffer_encoder(kp, buffer_pos);
        imap[input_id * (kv - 1) + kp] = buffer_code;
        omap[output_id * (kv - 1) + kp] = buffer_code;
    }
}


__global__ void queryHash_wholemap_stride1(
                    const int nnz, 
                    const int size, 
                    const int *__restrict__ coord,
                    const int ks, 
                    const int kv, 
                    const uint64_t *val, 
                    const int *idx, 
                    int *imap,                 // input-major mapping, (nnz * (kv - 1))
                    int *omap,                 // output-major mapping, (nnz * (kv - 1))
                    int *knnz)                 // the counter of nnz for each each kernel offsets, (kv - 1)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;  // a thread for a nnz


    // exclude illegal id number
    if (nid < nnz)
    {
        // attention: bank conflicts
        // extern __shared__ int counter[];

        int Nx = coord[nid * 3];
        int Ny = coord[nid * 3 + 1];
        int Nz = coord[nid * 3 + 2];

        for (int k = 0; k < (kv - 1) / 2; k++){
            
            int kx = k / (ks * ks) - (ks - 1) / 2;
            int ky = (k / ks) % ks - (ks - 1) / 2;
            int kz = k % ks - (ks - 1) / 2;
            // int kx = kernel_offset(k, 0);
            // int ky = kernel_offset(k, 1);
            // int kz = kernel_offset(k, 2);

            int colli_num = 0; 

            int Tx = Nx + kx;
            int Ty = Ny + ky;
            int Tz = Nz + kz;
        
            uint64_t target_val = coord_hash(Tx, Ty, Tz);
            
            uint64_t target_id = target_val % (uint64_t)size;

            // find target or empty
            while (val[target_id] != target_val && idx[target_id] > -1){
                colli_num += 1;
                if (colli_num == COLLISION_BOUND){continue;}
                target_id = (target_id + 97) % size;
            }
            // set map = id or -1
            int id_to_write = idx[target_id];

            if(id_to_write < 0){continue;}

            int buffer_o_pos = atomicAdd(&knnz[k], 1);
            int buffer_o_code = buffer_encoder(k, buffer_o_pos);

            int buffer_i_pos = atomicAdd(&knnz[kv - 2 - k], 1);
            int buffer_i_code = buffer_encoder(kv - 2 - k, buffer_i_pos);

            // id_to_write -- k -- nid
            omap[nid * (kv - 1) + k] = buffer_o_code;
            imap[id_to_write * (kv - 1) + k] = buffer_o_code;     

            // nid -- ( kv - 2 - k ) -- id_to_write
            omap[id_to_write * (kv - 1) + kv - 2 - k] = buffer_i_code;
            imap[nid * (kv - 1) + kv - 2 - k] = buffer_i_code;
        }
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
    for (int k = 0; k < kv - 1; k++){
        bool effective = map[nid * (kv - 1) + k] >= 0;
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


/* Hand-written kernel is much slower than thrust ...*/
__global__ void exclusive_scan_kernel(
                const int nnz, 
                int *input, 
                int *output
){
    // a thread for a scan
    const int _STRIDE_NNZ_BLOCK = gridDim.x;
    const int nid = blockIdx.x + threadIdx.x * _STRIDE_NNZ_BLOCK;
    if(nid >= nnz){return;}
    int maxnwb = blockIdx.x + ((nnz - blockIdx.x - 1) 
         / _STRIDE_NNZ_BLOCK) * _STRIDE_NNZ_BLOCK;
    int minnwb = blockIdx.x;
    // assume _STRIDE_NNZ_BLOCK an even number
    int amount = (minnwb + maxnwb) / 2;
    // reading
    int acc1 = 0;
#pragma unroll
    for (int i = 0; i < min(nid, amount); i++){
        acc1 += input[i];
    }
    int acc2 = 0;
#pragma unroll
    for (int j = 0; j < amount - nid; j++){
        acc2 += input[j + amount];
    }
    // writing
    // possibly two threads are writing together
    atomicAdd(&output[nid], acc1);
    if (nid > amount){return;}
    atomicAdd(&output[maxnwb + minnwb - nid], acc2);
}


__global__ void prescan(
                const int n, 
                int *csr
){
    extern __shared__ int temp[];
    int thid = threadIdx.x; 
    int offset = 1;
    temp[2 * thid] = csr[2 * thid];
    for (int d = n>>1; d > 0; d >>= 1){
        __syncthreads();
        if (thid < d){
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1; 
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0) { temp[n - 1] = 0; } 
    for (int d = 1; d < n; d *= 2){
        offset >>= 1;
        __syncthreads();
        if (thid < d){
            int ai = offset*(2*thid+1)-1;  
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai]; 
            temp[ai] = temp[bi]; 
            temp[bi] += t; 
        }
    }
    __syncthreads(); 
    csr[2*thid] = temp[2*thid];
    csr[2*thid+1] = temp[2*thid+1]; 
}

