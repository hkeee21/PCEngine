//kernel
#include <cuda.h>
#include <stdlib.h>
#include <iostream>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

extern "C"
//#include <torch/types.h>

// 二分查找 + 计算nnz
__device__ float compute_nnz_2(const int Ax, const int Ay, const int row_A, const int col_A, const int* rowptr, const int* colind, const float* value,
                const int Bx, const int By, const int size_B, const float* kernel)
{
    int row = Ax + (Bx - (size_B/2));
    int col = Ay + (By - (size_B/2));
    // 防止越界
    if(row >= 0 && row < row_A && col >=0 && col < col_A)
    {
        int colstart = rowptr[row];
        int colend = rowptr[row + 1];
        while(colstart <= colend)
        {
            int mid = (colstart + colend) >> 1;
            if(colind[mid] > col)
                colend = mid - 1;
            else if(colind[mid] < col)
                colstart = mid + 1;          
            else
                return kernel[Bx*size_B + By] * value[mid];
        }
        return 0;
    }
    else
        return 0;
}

// find the position of BlockIdx.x
__device__ void find_position_2(const int row_A, const int col_A, const int* rowptr, const int* colind, const int id, int *Ax, int *Ay)
{
    int rowstart = 0;
    int rowend = row_A + 1;
    while(rowstart + 1 < rowend)
    {
        int mid = (rowstart + rowend) >> 1;
        if(rowptr[mid] > id)
            rowend = mid;
        else if(rowptr[mid] <= id)
            rowstart = mid;
    }
    if(id >= rowptr[rowstart] && id < rowptr[rowend])
    {
        *Ax = rowstart;
        *Ay = colind[id];
    }
    else   
        printf("can not find the position of BlockIdx\n");   
    
}

// // sparse conv
// __global__ void SimpleConv(const int size_A, const int* rowptr, const int* colind, const float* value, 
//                             const int size_B,const float* kernel, float* out) 
// {
//     extern __shared__ float sh[];
//     int Ax = blockIdx.x; // A矩阵的第Ax行
//     int colstart = rowptr[Ax];
//     if(blockIdx.y < rowptr[Ax+1] - rowptr[Ax])
//     {
//         int Ay = colind[colstart + blockIdx.y]; // A矩阵的第Ay列
//         if(threadIdx.x < size_B*size_B)
//         {
//             int Bx = threadIdx.x / size_B; // B卷积核的第Bx行
//             int By = threadIdx.x % size_B; // B卷积核的第By列
//             sh[Bx*size_B + By] = compute_nnz(Ax, Ay, size_A, rowptr, colind, value,
//                                             Bx, By, size_B, kernel);
//             __syncwarp();

//             // 待改进
//             // kernel乘积结果累加到中心node
//             for(int i = 0; i < size_B*size_B; i++)
//                 out[colstart + blockIdx.y] += sh[i]; 
//         }
//     }  
// }

// __global__ void MidConv(const int size_A, const int* rowptr, const int* colind, const float* value, 
//                         const int size_B,const float* kernel, float* out) 
// {
//     extern __shared__ float sh[];
//     int Ax = blockIdx.x; // A矩阵的第Ax行
//     int colstart = rowptr[Ax];
//     if(blockIdx.y < rowptr[Ax + 1] - rowptr[Ax])
//     {
//         int Ay = colind[colstart + blockIdx.y]; // A矩阵的第Ay列
//         if(threadIdx.x < size_B && threadIdx.y < size_B)
//         {
//             int Bx = threadIdx.x; // B卷积核的第Bx行
//             int By = threadIdx.y; // B卷积核的第By列
//             sh[Bx*size_B + By] = compute_nnz(Ax, Ay, size_A, rowptr, colind, value,
//                                             Bx, By, size_B, kernel);
//             __syncwarp();

//             // 待改进
//             // kernel乘积结果累加到中心node
//             for(int i = 0; i < size_B*size_B; i++)
//                 out[colstart + blockIdx.y] += sh[i];    
//         }
//     }  
// }

// sparse conv_2
__global__ void SimpleConv_2(const int row_A, const int col_A, const int* rowptr, const int* colind, const float* value, 
                                const int size_B,const float* kernel, float* out) 
{
    int nnz = rowptr[row_A];
    int id = blockIdx.x; // A矩阵的第Ax行
    int Ax, Ay;
    extern __shared__ float sh[];
    while(id < nnz)
    {
        if(threadIdx.x < size_B*size_B)
        {
            
            find_position_2(row_A, col_A, rowptr, colind, id, &Ax, &Ay);
            int Bx = threadIdx.x / size_B; // B卷积核的第Bx行
            int By = threadIdx.x % size_B; // B卷积核的第By列
            sh[Bx*size_B + By] = compute_nnz_2(Ax, Ay, row_A, col_A, rowptr, colind, value,
                                            Bx, By, size_B, kernel);
        }
        __syncwarp();
        // 待改进
        // kernel乘积结果累加到中心node
        // for(int i = 0; i < size_B*size_B; i++)
        //     out[id] += sh[i];

        for(int step = 16; step > 0; step = (step >> 1))
        {
            if(threadIdx.x < step)
            {   
                sh[threadIdx.x] += sh[threadIdx.x + step];
            }
        }
        if(threadIdx.x == 0)
        {   
            out[id] = sh[0]; 
        }
        id += gridDim.x;
    }
    
}

__global__ void MidConv_2(const int row_A, const int col_A, const int* rowptr, const int* colind, const float* value, 
                            const int size_B,const float* kernel, float* out) 
{
    int nnz = rowptr[row_A];
    int id = blockIdx.x; // A矩阵的第Ax行
    int Ax, Ay;
    extern __shared__ float sh[];
    while(id < nnz)
    { 
        if(threadIdx.x < size_B && threadIdx.y < size_B)
        {   
            
            find_position_2(row_A, col_A, rowptr, colind, id, &Ax, &Ay);
            int Bx = threadIdx.x; // B卷积核的第Bx行
            int By = threadIdx.y; // B卷积核的第By列
            sh[Bx*size_B + By] = compute_nnz_2(Ax, Ay, row_A, col_A, rowptr, colind, value,
                                             Bx, By, size_B, kernel);
        } 
        __syncwarp();
        // 待改进，有错误！
        // kernel乘积结果累加到中心node
        for(int i = 0; i < size_B*size_B; i++)
            out[id] += sh[i];   

        id += gridDim.x;
    }
}

// void sparse_conv(const int size_A, const int* rowptr, const int* colind, const float* value, const int size_B,
//     const float* kernel, float* out) 
//     {
//     int nnz_per_row = nnzmax_per_row(size_A, rowptr);
//     if (size_B < 6) {
//         SimpleConv
//             <<<dim3(size_A, nnz_per_row, 1), dim3(32, 1, 1)>>>(
//             size_A, rowptr, colind, value, size_B, kernel, out);
//     }
// }