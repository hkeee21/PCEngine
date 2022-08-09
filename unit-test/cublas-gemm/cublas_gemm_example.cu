/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"


void randomInit(float *data, int size)
{
    float density = 1;
    float s = 1/density;
    int ss = s;
    for (int i = 0; i < size; ++i)
    {
        if((int)rand()%ss == 0)
        {
            data[i] = rand() / (float)RAND_MAX;
        }
        else
            data[i] = 0;
    }
}


void GEMM_CPU(const int kn, const int c_in, const int c_out,
                const float *in_f, const float *kv, 
                float *out)
{
    for (int i = 0; i < kn; i++){
        for (int co = 0; co < c_out; co++){
            float tv = 0;
            for (int c = 0; c < c_in; c++){
                tv += kv[co * c_in + c] * in_f[c * kn + i];
            }
            out[co * kn + i] += tv;
        }
    }
    printf("Computation on CPU Done.\n");  
}


float CheckResults(const int len, const int c_out, const float *cpu_results, const float *gpu_results){
    float accum_error = 0;
    for (int i = 0; i < len; i++){
        int n = i / c_out;
        int c = i % c_out;
        float error = fabs(cpu_results[i] - gpu_results[i]);
        if (error > 1.0e-3f){
            printf("The %d-th nnz's %d-th channel has abs error: %f\n", n, c, error);
        }
        accum_error += error;
    }

    return accum_error;
}


void TransMat(const int row, const int col, const float *M, float *tM){
    for (int r = 0; r < row; r++){
        for (int c = 0; c < col; c++){
            tM[c * row + r] = M[r * col + c];
        }
    }
}

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 5080;   // kernel_nnz
    const int n = 5080;   // out_channel
    const int k = 5080;   // in_channel

    const int lda = m;    // leading dimension of a
    const int ldb = k;
    const int ldc = m;

    int iter_num = 100;

    float *A = (float *)malloc(m * k * sizeof(float));
    randomInit(A, m * k);

    float *B = (float *)malloc(k * n * sizeof(float));
    randomInit(B, k * n);

    float *C = (float *)malloc(m * n * sizeof(float));
    memset(C, 0, m * n * sizeof(float));

    const float alpha = 1.0;
    const float beta = 0.0;

    float *c_C = (float *)malloc(m * n * sizeof(float));
    memset(c_C, 0, m * n * sizeof(float));

    // GEMM_CPU(m, k, n, A, B, c_C);


    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    /*const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    std::vector<data_type> C(m * n);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;*/

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /*printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");*/

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice, stream));

    /*
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));*/

    /* step 3: compute */
    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */

    /*
    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");*/

    // float gemm_error = CheckResults(m * n, n, c_C, C);
    // printf("The accumulated abs error: %f\n", gemm_error);


     /* free resources */
    free(A);
    free(B);
    free(C);
    free(c_C);


    // profiling
    // warm up
    for(int w = 0; w < 10; w++)
    {
        CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
        
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    // Allocate CUDA events that we'll use for timing
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, NULL));

    for(int i = 0; i < iter_num; i++)
    {
        CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
        
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaDeviceSynchronize();
    }

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerGemm = msecTotal / iter_num;
    double flopsPerGemm = 2.0 * (double)m * (double)n * (double)k;
    double GFlops = (flopsPerGemm * 1.0e-9f) / (msecPerGemm / 1000.0f);

    printf(
        "Performance= %.4f GFlop/s, Time= %.4f msec, Size= %.0f Ops\n",
        GFlops,
        msecPerGemm,
        flopsPerGemm);
    

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}