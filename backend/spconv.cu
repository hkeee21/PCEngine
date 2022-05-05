#include "spconv.h"

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

using namespace std;
#define BLOCK_SIZE 32

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)

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


void coordInit(int *coord, int n)
{
    int Vx = 0;
    int Vy = 0;
    int Vz = 0;
    for (int i = 0; i < n; i++){
        coord[3*i] = Vx;
        coord[3*i+1] = Vy;
        coord[3*i+2] = Vz;
        if(i % 10 == 0){Vx = Vx + 3; Vy = 0; Vz = 0;}
        if(i % 5 == 4){Vy = Vy + 2; Vz = 0;}
        Vz = Vz + 1;
    }
}


int main(void){

    printf("[SubmanifoldSparseConv] - Starting...\n");

    int nnz = 32;
    int in_channel = 3;
    int out_channel = 3;
    int k_size = 3;
    int k_vol = k_size * k_size * k_size;

    size_t const blocknum = (nnz + BLOCK_SIZE  - 1) / BLOCK_SIZE;
    size_t const gridnum = nnz > out_channel ? (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE : (out_channel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // initialize weights and inputs
    printf("Initialize the weights and inputs ...\n");
    float *weights = (float *)malloc(k_vol * in_channel * out_channel * sizeof(float));
    randomInit(weights, k_vol * in_channel * out_channel);

    float *in_feats = (float *)malloc(nnz * in_channel * sizeof(float));
    randomInit(in_feats, nnz * in_channel);
    
    /*for (int i = 0; i < nnz ; i++){
        printf("[%.4f,%.4f,%.4f]\n", in_feats[3 * i], in_feats[3 * i + 1], in_feats[3 * i + 2]);
    }*/

    int *in_coords = (int *)malloc(nnz * 3 * sizeof(int));
    coordInit(in_coords, nnz);

    /*for (int i = 0; i < nnz ; i++){
        printf("[%d,%d,%d]\n", in_coords[3 * i], in_coords[3 * i + 1], in_coords[3 * i + 2]);
    }*/

    // allocate memory for outputs
    float *out_feats = (float *)malloc(nnz * out_channel * sizeof(float));

    float *out_cpu = (float *)malloc(nnz * out_channel * sizeof(float));

    float *out_gemm_cpu = (float *)malloc(nnz * out_channel * sizeof(float));

    // CPU Conv
    Conv_CPU(nnz, in_channel, out_channel, in_coords, in_feats, weights, k_size, out_cpu);
    
    /*for (int i = 0; i < nnz ; i++){
        printf("[%.4f,%.4f,%.4f]\n", out_cpu[3 * i], out_cpu[3 * i + 1], out_cpu[3 * i + 2]);
    }*/

    // GPU memory
    float *dev_weights, *dev_infeats, *dev_outfeats;
    int *dev_incoords;
    int *dev_inmap;
    
    int *in_map = (int *)malloc(nnz * sizeof(int));
    int *map_cpu = (int *)malloc(nnz * sizeof(int));

    // TODO: move the mem allocation of weights into the iteration
    checkCudaError(cudaMalloc((void **)&dev_weights, k_vol * in_channel * out_channel * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&dev_infeats, nnz * in_channel * sizeof(float)));
    checkCudaError(cudaMalloc((void **)&dev_outfeats, nnz * out_channel * sizeof(float)));

    checkCudaError(cudaMalloc((void **)&dev_incoords, nnz * 3 * sizeof(int)));
    checkCudaError(cudaMalloc((void **)&dev_inmap, nnz * sizeof(int)));

    checkCudaError(cudaMemcpy(dev_weights, weights, k_vol * in_channel * out_channel * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_infeats, in_feats, nnz * in_channel * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_incoords, in_coords, nnz * 3 * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemset(dev_outfeats, 0, nnz * out_channel * sizeof(float)));

    // GPU Conv
    for (int i = 0; i < k_vol; i++){

        checkCudaError(cudaMemset(dev_inmap, -1, nnz * sizeof(int)));
        // checkCudaError(cudaMemcpy(map_cpu, in_map, nnz * sizeof(int), cudaMemcpyDeviceToHost));
        memset(map_cpu, -1, nnz * sizeof(int));
        /*for (int j = 0; j < nnz ; j++){
            printf("%d\n", map_cpu[j]);
        }*/

        int k_offset_x = i / (k_size * k_size) - (k_size - 1) / 2;
        int k_offset_y = (i / k_size) % k_size - (k_size - 1) / 2;
        int k_offset_z = i % k_size - (k_size - 1) / 2;

        printf("Handling the %d-th kernel offset ...\n", i+1);
        printf("The offset is (%d, %d, %d).\n", k_offset_x, k_offset_y, k_offset_z);

        search<<<dim3(blocknum, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(
                nnz, 
                in_channel, out_channel,
                dev_incoords, 
                k_offset_x, k_offset_y, k_offset_z, 
                dev_inmap);

        // search_CPU(nnz, in_channel, out_channel, in_coords, k_offset_x, k_offset_y, k_offset_z, map_cpu);
        /*for (int j = 0; j < nnz ; j++){
             printf("%d\n", map_cpu[j]);
        }*/

        printf("Map search is done.\n");
        checkCudaError(cudaMemcpy(in_map, dev_inmap, nnz * sizeof(int), cudaMemcpyDeviceToHost));
        // for (int j = 0; j < nnz ; j++){
        //      printf("%d - %d\n", in_map[j], j);
        // }
        
        gemm<<<dim3(gridnum, gridnum, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(
                nnz, 
                in_channel, out_channel,
                dev_infeats,
                &dev_weights[i * in_channel * out_channel],
                dev_outfeats,
                dev_inmap);
        
        printf("GEMM is done.\n");

        gemm_cpu(nnz, in_channel, out_channel, in_feats, &weights[i * in_channel * out_channel], in_map, out_gemm_cpu);

    }
    printf("All computation is done.\n");

    checkCudaError(cudaMemcpy(out_feats, dev_outfeats, nnz * out_channel * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Output is copied.\n");

    // Compare with CPU results
    for (int i = 0; i < nnz ; i++){
        printf("[%.4f,%.4f,%.4f] - [%.4f,%.4f,%.4f] - [%.4f,%.4f,%.4f]\n", 
        out_cpu[3 * i], out_cpu[3 * i + 1], out_cpu[3 * i + 2],
        out_feats[3 * i], out_feats[3 * i + 1], out_feats[3 * i + 2], 
        out_gemm_cpu[3 * i], out_gemm_cpu[3 * i + 1], out_gemm_cpu[3 * i + 2]);
    }
    

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    free(weights);
    free(in_feats);
    free(in_coords);
    free(out_feats);
    free(map_cpu);
    free(in_map);
    checkCudaError(cudaFree(dev_weights));
    checkCudaError(cudaFree(dev_infeats));
    checkCudaError(cudaFree(dev_incoords));
    checkCudaError(cudaFree(dev_outfeats));
    checkCudaError(cudaFree(dev_inmap));

    return 0;
}