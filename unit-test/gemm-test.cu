#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>

using namespace std;

#define BLOCK_SIZE 16


extern "C"

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)


void GEMM_CPU(const int kn, const int c_in, const int c_out,
                const int *in_idx, const int *out_idx, 
                const float *in_f, const float *kv, 
                float *out)
{
    for (int i = 0; i < kn; i++){
        int in_row = in_idx[i];
        int out_row = out_idx[i];
        for (int co = 0; co < c_out; co++){
            float tv = 0;
            for (int c = 0; c < c_in; c++){
                tv += kv[c * c_out + co] * in_f[in_row * c_in + c];
            }
            out[out_row * c_out + co] += tv;
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


__global__ void gemm(const int nnz, const int kernel_nnz, const int c_in, const int c_out,
                const float *__restrict__ in_f, const float *__restrict__ kv, float *out_f,
                const int *in_idx, const int *out_idx) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Coordinate. x is for rows, y is for columns.
  const int x = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * by + ty;
  // const int y = BLOCK_SIZE * bx + tx;
  // const int x = BLOCK_SIZE * by + ty;

  // The thread deals with the x-th channel of the y-th output
  const int in_row = y < kernel_nnz ? in_idx[y] : -1;
  const int out_row = y < kernel_nnz ? out_idx[y] : -1;

  if(in_row > -1 && out_row > -1){
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
    As[ty][tx] = ((s + tx) < c_in && in_row < nnz) ? in_f[c_in * in_row + s + tx] : 0;
    Bs[ty][tx] = ((s + ty) < c_in && x < c_out) ? kv[c_out * (s + ty) + x] : 0;

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
  if (out_row < nnz && x < c_out)
    atomicAdd(&out_f[c_out * out_row + x], Csub);
  // C[wB * out_row + x] += Csub;
  }
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


void indexInit(int *index, int n, int kn)
{
    for (int i = 0; i < kn; i++){
        index[i] = rand() % n;  // 0 ~ n-1 for index
    }
}


int main(void){

    int nnz = 100000;
    int kernel_nnz = 100000;
    int in_channel = 256;
    int out_channel = 512;

    int iter_num = 100;


    // initialize weights and inputs
    printf("Initialize the weights and inputs ...\n");
    float *weights = (float *)malloc(in_channel * out_channel * sizeof(float));
    randomInit(weights, in_channel * out_channel);

    float *in_feats = (float *)malloc(nnz * in_channel * sizeof(float));
    randomInit(in_feats, nnz * in_channel);

    // initialize indeces for inputs and outputs
    int *in_index = (int *)malloc(kernel_nnz * sizeof(int));
    indexInit(in_index, nnz, kernel_nnz);

    int *out_index = (int *)malloc(kernel_nnz * sizeof(int));
    indexInit(out_index, nnz, kernel_nnz);

    // allocate memory for outputs
    float *out_feats = (float *)malloc(nnz * out_channel * sizeof(float));
    memset(out_feats, 0, nnz * out_channel * sizeof(float));

    float *out_cpu = (float *)malloc(nnz * out_channel * sizeof(float));
    memset(out_cpu, 0, nnz * out_channel * sizeof(float));


    // CPU gemm
    GEMM_CPU(kernel_nnz, in_channel, out_channel, in_index, out_index, in_feats, weights, out_cpu);


    // GPU memory
    float *dev_weights, *dev_infeats, *dev_outfeats;
    int *dev_in, *dev_out;

    checkCudaError(cudaMalloc((void **)&dev_weights, in_channel * out_channel * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&dev_infeats, nnz * in_channel * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&dev_outfeats, nnz * out_channel * sizeof(float)));

    checkCudaError(cudaMalloc((void **)&dev_in, kernel_nnz * sizeof(int)));
    checkCudaError(cudaMalloc((void **)&dev_out, kernel_nnz * sizeof(int)));
    
    checkCudaError(cudaMemcpy(dev_weights, weights, in_channel * out_channel * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_infeats, in_feats, nnz * in_channel * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_in, in_index, kernel_nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_out, out_index, kernel_nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemset(dev_outfeats, 0, nnz * out_channel * sizeof(float)));
    

    // gemm kernel
    // check if the results match
    size_t const gridnum_x = (out_channel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t const gridnum_y = (kernel_nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gemm<<<dim3(gridnum_x, gridnum_y), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(
                nnz, 
                kernel_nnz, 
                in_channel, out_channel,
                dev_infeats,
                dev_weights,
                dev_outfeats,
                dev_in,
                dev_out);

    printf("GPU gemm is done.\n");

    checkCudaError(cudaMemcpy(out_feats, dev_outfeats, nnz * out_channel * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Output is copied.\n");

    float gemm_error = CheckResults(nnz * out_channel, out_channel, out_cpu, out_feats);
    printf("The accumulated abs error: %f\n", gemm_error);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    free(weights);
    free(in_feats);
    free(in_index);
    free(out_index);
    free(out_feats);
    free(out_cpu);


    // profiling
    cudaEvent_t start, stop;
    // Allocate CUDA events that we'll use for timing
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // Record the start event
    checkCudaError(cudaEventRecord(start, NULL));

    for(int i = 0; i < iter_num; i++)
    {
        gemm<<<dim3(gridnum_x, gridnum_y), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(
                nnz, 
                kernel_nnz, 
                in_channel, out_channel,
                dev_infeats,
                dev_weights,
                dev_outfeats,
                dev_in,
                dev_out);
    }


    // Record the stop event
    checkCudaError(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaError(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaError(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerGemm = msecTotal / iter_num;
    double flopsPerGemm = 2.0 * (double)kernel_nnz * (double)in_channel * (double)out_channel;
    double GFlops = (flopsPerGemm * 1.0e-9f) / (msecPerGemm / 1000.0f);

    printf(
        "Performance= %.4f GFlop/s, Time= %.4f msec, Size= %.0f Ops\n",
        GFlops,
        msecPerGemm,
        flopsPerGemm);


    checkCudaError(cudaFree(dev_weights));
    checkCudaError(cudaFree(dev_infeats));
    checkCudaError(cudaFree(dev_in));
    checkCudaError(cudaFree(dev_out));
    checkCudaError(cudaFree(dev_outfeats));
    

    return 0;
}