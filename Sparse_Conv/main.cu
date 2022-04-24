// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <stdlib.h>
//#include <iostream>
#include <math.h>
#include <stdio.h>

#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include "spconv.h"
#include "util.hpp"

using namespace std;

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)


// sparse conv on cpu
void Conv_CPU(const int size_A, float* h_A, 
                const int size_B, float* h_B, float* out)
{
    printf("compute on cpu start... \t");
    for(int i = 0; i < size_A; ++i){
        for(int j = 0; j < size_A; ++j){
            float sum = 0;
            if(h_A[i*size_A + j] != 0){  // 默认值为负数
                for(int k = 0; k < size_B*size_B; ++k){   
                    int Bx = k / size_B;
                    int By = k % size_B;
                    int row = i + (Bx - (size_B/2));
                    int col = j + (By - (size_B/2)); 
                    if(row >= 0 && row < size_A && col >=0 && col < size_A)
                    {
                        sum += h_A[row*size_A + col] * h_B[Bx*size_B + By];
                    }
                }
                out[i*size_A + j] = sum;
            }
            else{
                out[i*size_A + j] = 0;
            }      
        }
    }
    printf("done \n");
}

// print the diff between gpu and cpu
void printDiff(const int size_A, const int* rowptr, const int* colind, 
                const float* out, const float* out_cpu, const int iListLength, const float fListTol)
{
    printf("print Diff...\n");
    for(int i = 0; i < size_A; ++i)
    {
        int lb = rowptr[i];
        int hb = rowptr[i+1];
        for(int j = lb; j < hb; ++j)
        {
            float fDiff = fabs(out[j] - out_cpu[i*size_A + colind[j]]);
            if(fDiff > fListTol)
            {   
                printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, out[j], out_cpu[i*size_A + colind[j]], fDiff);
            }
        }
    }
    printf("done\n");
}

void initializeCUDA(int argc, char **argv, int &devID)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
}

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

int countnnz(float* h_A, int size_A)
{
    int sum = 0;
    for(int i = 0; i < size_A; i++)
    {  
        for(int j = 0; j < size_A; j++)
        {
            if(h_A[i * size_A + j] != 0) // 默认值为负数
                sum++;
        }
    }
    return sum;
}

void ToCSR(const float* h_A, const int size_A, int* rowptr, int* colind, 
            float* value)
{
    int sum = 0;
    rowptr[0] = 0;
    for(int i = 0; i < size_A; i++)
    {  
        for(int j = 0; j < size_A; j++)
        {
            if(h_A[i*size_A + j] != 0) // 默认值为负数
            {
                colind[sum] = j;
                value[sum] = h_A[i*size_A + j];
                sum++;
            }       
        }
        rowptr[i+1] = sum;
    }
}

// count max nnz_per_row，用于blockDim.y
int nnzmax_per_row(const int size_A, const int* rowptr)
{
    int max = 0;
    for(int i = 0; i < size_A; i++)
    {  
        max = ((rowptr[i+1] - rowptr[i]) > max) ? (rowptr[i+1] - rowptr[i]) : max; 
    }
    return max;
}

void print_cpu(int size_A, float* out_cpu)
{
    printf("result of cpu:\n");
    for(int i = 0; i < size_A; i++)
    {
        for(int j = 0; j < size_A; j++)
        {
            printf("%.2f\t", out_cpu[i*size_A + j]);
        }
        printf("\n");
    }
}

void print_cpu_2(int nnz, float* out)
{
    printf("result of cpu:\n");
    for(int i = 0; i < nnz; i++)
    {
        printf("%.2f\t", out[i]); 
    }
    printf("\n");
}

void print_gpu(int nnz, float* out)
{
    printf("result of gpu:\n");
    for(int i = 0; i < nnz; i++)
    {
        printf("%.2f\t", out[i]); 
    }
    printf("\n");
}

// int matrixConv(int argc, char **argv, int devID, const int size_A, const int size_B)
// {
//     cudaDeviceProp deviceProp;

//     checkCudaError(cudaGetDeviceProperties(&deviceProp, devID));

//     // set seed for rand()
//     srand(2006);

//     // allocate host memory for matrices A and B
//     int mem_size_A = sizeof(float) * size_A * size_A;
//     float *h_A = (float *)malloc(mem_size_A);
//     int mem_size_B = sizeof(float) * size_B * size_B; 
//     float *h_B = (float *)malloc(mem_size_B);

//     float *out_cpu = (float*)malloc(mem_size_A);

//     // initialize host memory
//     randomInit(h_A, size_A*size_A);
//     randomInit(h_B, size_B*size_B);

//     // compute on CPU
//     Conv_CPU(size_A, h_A, size_B, h_B, out_cpu);

//     //count nnz of A
//     int nnz = countnnz(h_A, size_A);

//     // initialize CSR memory
//     int mem_size_rowptr = sizeof(int) * (size_A + 1);
//     int mem_size_colind = sizeof(int) * nnz;
//     int mem_size_value = sizeof(float) * nnz;
//     int *rowptr = (int*)malloc(mem_size_rowptr);
//     int *colind = (int*)malloc(mem_size_colind);
//     float *value = (float*)malloc(mem_size_value);

//     ToCSR(h_A, size_A, rowptr, colind, value);

//     // allocate device memory
//     int *d_rowptr, *d_colind;
//     float *d_value, *d_B, *d_out;

//     // allocate host memory for the result
//     float *out = (float*)malloc(mem_size_value);
    
//     checkCudaError(cudaMalloc((void **) &d_rowptr, mem_size_rowptr));
//     checkCudaError(cudaMalloc((void **) &d_colind, mem_size_colind));
//     checkCudaError(cudaMalloc((void **) &d_value, mem_size_value));
//     checkCudaError(cudaMalloc((void **) &d_B, mem_size_B));
    

//     checkCudaError(cudaMemcpy(d_rowptr, rowptr, mem_size_rowptr, cudaMemcpyHostToDevice));
//     checkCudaError(cudaMemcpy(d_colind, colind, mem_size_colind, cudaMemcpyHostToDevice));
//     checkCudaError(cudaMemcpy(d_value, value, mem_size_value, cudaMemcpyHostToDevice));
//     checkCudaError(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

//     // allocate device memory for out
//     checkCudaError(cudaMalloc((void **) &d_out, mem_size_value));
//     checkCudaError(cudaMemset(d_out, 0, mem_size_value));//设置初始值为0

//     //int nnz_per_row = nnzmax_per_row(size_A, rowptr);

//     cudaEvent_t start, stop;
//     // Allocate CUDA events that we'll use for timing
//     checkCudaError(cudaEventCreate(&start));
//     checkCudaError(cudaEventCreate(&stop));

//     // Record the start event
//     checkCudaError(cudaEventRecord(start, NULL));

//     // size_B <= 32
//     if(size_B < 6) {
//         SimpleConv_2
//             <<<dim3(nnz, 1, 1), dim3(32, 1, 1), sizeof(float)*32>>>(
//             size_A, d_rowptr, d_colind, d_value, size_B, d_B, d_out);
//     }
//     // size_B <= 1024
//     else if(size_B <= 32)
//     {
//         MidConv_2
//             <<<dim3(nnz, 1, 1), dim3(32, 32, 1), sizeof(float)*1024>>>(
//             size_A, d_rowptr, d_colind, d_value, size_B, d_B, d_out);
//     }

//     // Record the stop event
//     checkCudaError(cudaEventRecord(stop, NULL));

//     // Wait for the stop event to complete
//     checkCudaError(cudaEventSynchronize(stop));

//     float msecTotal = 0.0f;
//     checkCudaError(cudaEventElapsedTime(&msecTotal, start, stop));

//     // copy result from device to host
//     checkCudaError(cudaMemcpy(out, d_out, sizeof(float) * nnz, cudaMemcpyDeviceToHost));

//     // check the result with cpu
//     printDiff(size_A, rowptr, colind, out, out_cpu, 100, 1.0e-5f);

//     // print the result
//     print_cpu(size_A, out_cpu);
//     print_gpu(nnz, out);

//     // clean up memory
//     free(h_A);
//     free(h_B);
//     free(rowptr);
//     free(colind);
//     free(value);
//     free(out);
//     free(out_cpu);
//     checkCudaError(cudaFree(d_rowptr));
//     checkCudaError(cudaFree(d_colind));
//     checkCudaError(cudaFree(d_value));
//     checkCudaError(cudaFree(d_B));
//     checkCudaError(cudaFree(d_out));

//     return 1;
// }

void COO_To_CSR(vector<int>& row_COO, vector<int>& col_COO, vector<float>& values_COO, 
                int S_mrows, int S_ncols, int nnz, int* rowptr, int* colind, float* value)
{
    rowptr[0] = 0;
    int sum = 0;
    for(int i = 0; i < S_mrows; i++)
    {
        while(row_COO[sum] == i)
        {
            sum++;
        }
        rowptr[i + 1] = sum;
    }
    if(rowptr[S_mrows] != nnz)
    {
        printf("Error in COO_To_CSR!\n");
    }
    // for(int i = 0; i < nnz; i++)
    // {
    //     colind[i] = col_COO[i];
    //     value[i] = values_COO[i];
    // }
}

void compute_cpu_COO(vector<int>& row_COO, vector<int>& col_COO, vector<float>& values_COO, 
                        int S_mrows, int S_ncols, int nnz, int size_B, float* h_B, float*out_cpu)
{
    printf("compute on cpu(COO format) start... \t");
    for(int i = 0; i < nnz; ++i){
        int Ax = row_COO[i];
        int Ay = col_COO[i];
        float sum = 0;
        for(int j = 0; j < nnz; j++)
        {
            if(row_COO[j] >= Ax - (size_B/2) && row_COO[j] <= Ax + (size_B/2) && col_COO[j] >= Ay - (size_B/2) && col_COO[j] <= Ay + (size_B/2))
            {
                int Bx = (row_COO[j] - Ax) + size_B/2;
                int By = (col_COO[j] - Ay) + size_B/2;
                sum += values_COO[j] * h_B[Bx*size_B + By];
            }
            else if(row_COO[j] > Ax + (size_B/2))
                break;
        }
        out_cpu[i] = sum;
    }
    printf("done \n");
}

void printDiff_2(float* out, float* out_cpu, int nnz)
{
    printf("print Diff...\n");
    int count = 0;
    for(int i = 0; i < nnz; ++i)
    {
        float fDiff = fabs(out[i] - out_cpu[i]);
        if(fDiff > 1.0e-5f)
        {   
            count++;
            printf("error at %d : %.2f\n", i, fDiff);
        }
    }
    printf("the number of errors is : %d\n", count);
    printf("done\n");
}

int matrixConv_2(vector<int>& row_COO, vector<int>& col_COO, vector<float>& values_COO, 
                    int S_mrows, int S_ncols, int nnz, int size_B)
{
    int mem_size_B = sizeof(float) * size_B * size_B; 
    float *h_B = (float *)malloc(mem_size_B);

    // initialize kernel memory
    randomInit(h_B, size_B*size_B);

    int mem_size_out_cpu = sizeof(float) * nnz;
    float *out_cpu = (float*)malloc(mem_size_out_cpu);

    compute_cpu_COO(row_COO, col_COO, values_COO, S_mrows, S_ncols, nnz, size_B, h_B, out_cpu);

    // allocate memory for CSR format
    int mem_size_rowptr = sizeof(int) * (S_mrows + 1);
    int mem_size_colind = sizeof(int) * nnz;
    int mem_size_value = sizeof(float) * nnz;
    int *rowptr = (int*)malloc(mem_size_rowptr);
    int *colind = (int*)malloc(mem_size_colind);
    float *value = (float*)malloc(mem_size_value);

    // COO TO CSR
    memcpy(colind, &col_COO[0], col_COO.size() * sizeof(col_COO[0]));
    memcpy(value, &values_COO[0], values_COO.size() * sizeof(values_COO[0]));

    COO_To_CSR(row_COO, col_COO, values_COO, S_mrows, S_ncols, nnz, rowptr, colind, value);

    int *d_rowptr, *d_colind;
    float *d_value, *d_B, *d_out;

    // allocate host memory for the result
    float *out = (float*)malloc(mem_size_value);
    
    checkCudaError(cudaMalloc((void **) &d_rowptr, mem_size_rowptr));
    checkCudaError(cudaMalloc((void **) &d_colind, mem_size_colind));
    checkCudaError(cudaMalloc((void **) &d_value, mem_size_value));
    checkCudaError(cudaMalloc((void **) &d_B, mem_size_B));
    

    checkCudaError(cudaMemcpy(d_rowptr, rowptr, mem_size_rowptr, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_colind, colind, mem_size_colind, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_value, value, mem_size_value, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // allocate device memory for out
    checkCudaError(cudaMalloc((void **) &d_out, mem_size_value));
    checkCudaError(cudaMemset(d_out, 0, mem_size_value));//设置初始值为0

    // size_B <= 32
    if(size_B < 6) {
        SimpleConv_2
            <<<dim3(nnz, 1, 1), dim3(32, 1, 1), sizeof(float)*32>>>(
                S_mrows, S_ncols, d_rowptr, d_colind, d_value, size_B, d_B, d_out);
    }
    // size_B <= 1024
    else if(size_B <= 32)
    {
        MidConv_2
            <<<dim3(nnz, 1, 1), dim3(32, 32, 1), sizeof(float)*32*32>>>(
                S_mrows, S_ncols, d_rowptr, d_colind, d_value, size_B, d_B, d_out);
    }

    cudaEvent_t start, stop;
    // Allocate CUDA events that we'll use for timing
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    int nIter = 30;
    // Record the start event
    checkCudaError(cudaEventRecord(start, NULL));

    for(int i = 0; i < nIter; i++)
    {
        checkCudaError(cudaMemset(d_out, 0, mem_size_value));//设置初始值为0
        // size_B <= 32
        if(size_B < 6) {
            SimpleConv_2
                <<<dim3(nnz, 1, 1), dim3(32, 1, 1), sizeof(float)*32>>>(
                    S_mrows, S_ncols, d_rowptr, d_colind, d_value, size_B, d_B, d_out);
        }
        // size_B <= 1024
        else if(size_B <= 32)
        {
            MidConv_2
                <<<dim3(nnz, 1, 1), dim3(32, 32, 1), sizeof(float)*32*32>>>(
                    S_mrows, S_ncols, d_rowptr, d_colind, d_value, size_B, d_B, d_out);
        }
    }
    

    // Record the stop event
    checkCudaError(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaError(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaError(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)nnz * (double)size_B * (double)size_B;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("The rows of A is: %d\n", S_mrows);
    printf("The cols of A is: %d\n", S_ncols);
    printf("The nnz of A is: %d\n", nnz);
    printf("The rows and cols of B(kernel) is: %d\n", size_B);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);

    // copy result from device to host
    checkCudaError(cudaMemcpy(out, d_out, sizeof(float) * nnz, cudaMemcpyDeviceToHost));

    printDiff_2(out, out_cpu, nnz);
    // print the result
    // print_cpu_2(nnz,out_cpu);
    // print_gpu(nnz, out);

    // clean up memory
    free(h_B);
    free(rowptr);
    free(colind);
    free(value);
    free(out);
    free(out_cpu);
    checkCudaError(cudaFree(d_rowptr));
    checkCudaError(cudaFree(d_colind));
    checkCudaError(cudaFree(d_value));
    checkCudaError(cudaFree(d_B));
    checkCudaError(cudaFree(d_out));

    return 1;

}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[SubmanifoldSparseConv] - Starting...\n");

    int devID = 0;
    int size_B = 5;
    //int size_A = 5;
    std::vector<int> row_COO;
    std::vector<int> col_COO;
    std::vector<float> values_COO;
    int S_mrows;
    int S_ncols;
    int nnz;
    readMtx<float>(argv[1], row_COO, col_COO, values_COO, S_mrows, S_ncols, nnz);
    
    initializeCUDA(argc, argv, devID);

    //int matrix_result = matrixConv(argc, argv, devID, size_A, size_B);

    int matrix_result = matrixConv_2(row_COO, col_COO, values_COO, S_mrows, S_ncols, nnz, size_B);

    return matrix_result;
}