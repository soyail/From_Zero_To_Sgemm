#pragma once


#include "gemm_gpu_mem_coalesce.h"

__global__ void gemm_gpu_mem_coalesce_kernel(
    const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C
){
    const int cRow = blockIdx.x * blockDim.x + threadIdx.y;
    const int cCol = blockIdx.y * blockDim.y + threadIdx.x;
    if(cRow<m && cCol<n){
        float tmp = 0;
        for(int l=0; l<k; ++l){
            tmp+= A[cRow*k+l] * B[l*n+cCol];
        }
        C[cRow*n+cCol] = alpha*tmp+beta*C[cRow*n+cCol];

    }
      
}

void gemm_gpu_mem_coalesce(
    const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C,
    cublasHandle_t handle
){
    dim3 gridDim(ceil(n/32), ceil(m/32));
    dim3 blockDim(32, 32);
    gemm_gpu_mem_coalesce_kernel<<<gridDim,blockDim>>>(m,n,k,A,alpha,B,beta,C);
}