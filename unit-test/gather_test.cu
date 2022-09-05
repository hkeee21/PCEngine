#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>
#include <algorithm>

#include "spconv.cuh"

using namespace std;

#define DIV_UP(x, y) (x + y - 1) / y
#define NNZS_PER_BLOCK 4
#define KOFS_THREADS 4
#define CHNS_THREADS 32
#define MPNS_PER_BLOCK 64

extern "C"

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)


void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand() / (float)RAND_MAX;    
    }
}

void mapInit(int *map, int *counter, int nnz, int kv, int mps)
{
    int left = mps;
    int lower_bound = (kv - 1);
    int upper_bound = (kv - 1);
    std::vector<int> to_fill(kv - 1);
    std::iota (std::begin(to_fill), std::end(to_fill), 0);
    for (int i = 0; i < nnz; ++i){

        for (int k = 0; k < kv - 1; ++k){
            map[i * (kv - 1) + k] = -1;
        }

        upper_bound = min((kv - 2), left - (nnz - i - 1) * 3);
        lower_bound = max(3, left - (nnz - i - 1) * (kv - 2));
        if (left == 0){continue;}
        int knnz = rand() % (upper_bound - lower_bound + 1) + lower_bound;

        if (i == nnz - 1){
            knnz = left;
        }

        std::random_shuffle(to_fill.begin(), to_fill.end());
        for (int k = 0; k < knnz; ++k){
            int kofs = to_fill[k];
            int kpos = counter[kofs];
            map[i * (kv - 1) + k] = kofs * 1186111 + kpos;
            counter[kofs]++;
        }

        left -= knnz;
    }
}


void csrmapInit(int *map, int *csr, int *counter, int nnz, int kv, int mps)
{
    int left = mps;
    int lower_bound = (kv - 1);
    int upper_bound = (kv - 1);
    std::vector<int> to_fill(kv - 1);
    std::iota (std::begin(to_fill), std::end(to_fill), 0);
    csr[0] = 0;
    for (int i = 0; i < nnz; ++i){

        upper_bound = min((kv - 2), left - (nnz - i - 1) * 2);
        lower_bound = max(2, left - (nnz - i - 1) * (kv - 2));
        if (left == 0){continue;}
        int knnz = rand() % (upper_bound - lower_bound + 1) + lower_bound;

        if (i == nnz - 1){
            knnz = left;
        }

        std::random_shuffle(to_fill.begin(), to_fill.end());
        for (int k = 0; k < knnz; ++k){
            int kofs = to_fill[k];
            int kpos = counter[kofs];
            map[csr[i] + k] = kofs * 1186111 + kpos;
            counter[kofs]++;
        }

        left -= knnz;
        csr[i + 1] = csr[i] + knnz;
    }
}


int main(int argc, char **argv){

    // kernel configuration
    /*const int nnzs_per_block = 4;
    const int kofs_threads = 2;
    const int chns_threads = 32;
    int *p_nnzs_per_block = (int*)&nnzs_per_block;
    int *p_kofs_threads = (int*)&kofs_threads;
    int *p_chns_threads = (int*)&chns_threads;
    if (argc > 3){
        *p_nnzs_per_block = atoi(argv[1]);
        *p_kofs_threads = atoi(argv[2]);
        *p_chns_threads = atoi(argv[3]);
    }*/
    // printf("NNZS_PER_BLOCK=%d, KOFS_THREADS=%d, CHNS_THREADS=%d\n", 
    //     NNZS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS);

    // input setup
    // nnz : 2000 ~ 200000
    int nnz = 10000;
    // map size : depend on data sparsity
    int mps = 100000;
    // input channel : 16, 32, 64, 128, 192, 256, 512
    int chn = 16; // 64;
    // kernel size : 2, 3, 5
    int ks = 3;
    int kv = ks * ks * ks;
    // profiling iterations
    int iter_num = 100;

    if (argc > 5){
        nnz = atoi(argv[1]);
        mps = atoi(argv[2]);
        chn = atoi(argv[3]);
        ks = atoi(argv[4]);
        iter_num = atoi(argv[5]);
    }

    // feature initialization
    float *feature = (float *)malloc(nnz * chn * sizeof(float));
    randomInit(feature, nnz * chn);

    float *gfeats = (float *)malloc(mps * chn * sizeof(float));
    memset(gfeats, 0, mps * chn * sizeof(float));

    // mapping initialization
    int *imap = (int *)malloc(mps * sizeof(int));
    int *icsr = (int *)malloc((nnz + 1) * sizeof(int));
    int *knnz = (int *)malloc((kv - 1) * sizeof(int));
    csrmapInit(imap, icsr, knnz, nnz, kv, mps);

    int *kpos = (int *)malloc((kv - 1) * sizeof(int));
    for(int k = 0; k < kv - 1; ++k){
        if (k == 0){
            kpos[k] = 0;
            continue;
        }
        kpos[k] = knnz[k - 1] + kpos[k - 1];
        // printf("%d ", kpos[k]);
    }

    /*for (int i = 0; i < nnz; i++){
        for (int k = icsr[i]; k < icsr[i + 1]; k++){
            printf("%d-%d ", imap[k] / 1186111, imap[k] % 1186111);
        }
        printf("(%d)\n", icsr[i + 1]);
    }*/

    // device memory allocation
    float *dev_feat = nullptr;
    float *dev_gfeat = nullptr;
    int *dev_imap = nullptr;
    int *dev_icsr = nullptr;
    int *dev_kpos = nullptr;

    checkCudaError(cudaMalloc((void **)&dev_feat, nnz * chn * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&dev_gfeat, mps * chn * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&dev_imap, mps * sizeof(int)));
    checkCudaError(cudaMalloc((void **)&dev_icsr, (nnz + 1) * sizeof(int)));
    checkCudaError(cudaMalloc((void **)&dev_kpos, (kv - 1) * sizeof(int)));

    checkCudaError(cudaMemcpy(dev_feat, feature, nnz * chn * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_gfeat, gfeats, mps * chn * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_imap, imap, mps * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_icsr, icsr, (nnz + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_kpos, kpos, (kv - 1) * sizeof(int), cudaMemcpyHostToDevice));

    size_t const BLOCK_NUM = DIV_UP(nnz, NNZS_PER_BLOCK);
    // size_t const BLOCK_NUM = DIV_UP(mps, MPNS_PER_BLOCK);
    // int const NNZS_PER_BLOCK = toConst(nnzs_per_block, 0);
    // int const KOFS_THREADS = toConst(kofs_threads, 0);
    // int const CHNS_THREADS = toConst(chns_threads, 0);

    // warmup
    for (int i = 0; i < 20; i++){
        /*gather_all_input_major_csr_template2
            <NNZS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS>
            <<<BLOCK_NUM, dim3(CHNS_THREADS, KOFS_THREADS, NNZS_PER_BLOCK), 
            (NNZS_PER_BLOCK * (kv - 1) + 1) * sizeof(int)>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_icsr, dev_imap, dev_gfeat
        );*/
        /*gather_all_input_major
            <<<DIV_UP(nnz * chn, 256), 256>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_imap, dev_gfeat
        );*/
        /*gather_all_input_major_csr_balance
            <MPNS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS>
            <<<BLOCK_NUM, dim3(CHNS_THREADS, KOFS_THREADS)>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_icsr, dev_imap, dev_gfeat
        );*/
        gather_all_input_major_csr_template5
            <NNZS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS>
            <<<BLOCK_NUM, dim3(CHNS_THREADS, NNZS_PER_BLOCK, KOFS_THREADS)>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_icsr, dev_imap, dev_gfeat
        );
        /*gather_all_input_major_csr_template3
            <NNZS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS>
            <<<BLOCK_NUM, dim3(CHNS_THREADS, KOFS_THREADS, NNZS_PER_BLOCK)>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_icsr, dev_imap, dev_gfeat
        );*/
    }

    // profiling
    cudaEvent_t start, stop;
    // Allocate CUDA events that we'll use for timing
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // Record the start event
    checkCudaError(cudaEventRecord(start, NULL));

    for(int i = 0; i < iter_num; i++)
    {
        /*gather_all_input_major_csr_template2
            <NNZS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS>
            <<<BLOCK_NUM, dim3(CHNS_THREADS, KOFS_THREADS, NNZS_PER_BLOCK), 
            (NNZS_PER_BLOCK * (kv - 1) + 1) * sizeof(int)>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_icsr, dev_imap, dev_gfeat
        );*/
        /*gather_all_input_major
            <<<DIV_UP(nnz * chn, 256), 256>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_imap, dev_gfeat
        );*/
        gather_all_input_major_csr_template5
            <NNZS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS>
            <<<BLOCK_NUM, dim3(CHNS_THREADS, NNZS_PER_BLOCK, KOFS_THREADS)>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_icsr, dev_imap, dev_gfeat
        );
        /*gather_all_input_major_csr_template3
            <NNZS_PER_BLOCK, KOFS_THREADS, CHNS_THREADS>
            <<<BLOCK_NUM, dim3(CHNS_THREADS, KOFS_THREADS, NNZS_PER_BLOCK)>>>(
                nnz, kv, mps, dev_kpos, chn, dev_feat, dev_icsr, dev_imap, dev_gfeat
        );*/
    }

    // Record the stop event
    checkCudaError(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaError(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaError(cudaEventElapsedTime(&msecTotal, start, stop));

    /*printf(
        "Input Info: nnz = %d, channel = %d, map size = %d, kernel size = %d, Time= %.4f msec\n",
        nnz, chn, mps, ks, msecTotal / iter_num);*/
    
    /*printf(
        "%d %d %d %d %.4f\n",
        nnz, chn, mps, ks, msecTotal / iter_num);*/

    printf("%.4f\n", msecTotal / iter_num);

    checkCudaError(cudaFree(dev_feat));
    checkCudaError(cudaFree(dev_gfeat));
    checkCudaError(cudaFree(dev_imap));
    checkCudaError(cudaFree(dev_icsr));
    checkCudaError(cudaFree(dev_kpos));

    return 0;
}