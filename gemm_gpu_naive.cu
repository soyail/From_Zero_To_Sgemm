#include <cuda_runtime.h>
#include <cmath>
#include "gemm_gpu_naive.h"



__global__ void gemm_gpu_naive_kernel(
    const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C
){
    const int cRow = blockIdx.x * blockDim.x + threadIdx.x;
    const int cCol = blockIdx.y * blockDim.y + threadIdx.y;
    if(cRow<m && cCol<n){
        float tmp = 0.0;
        for(int l=0; l<k; ++l){
            tmp+= A[cRow*k+l] * B[l*n+cCol];
        }
        // C = alpha*A*B+beta*C
        C[cRow*n+cCol] = alpha*tmp+beta*C[cRow*n+cCol];

    }
}

void gemm_gpu_naive(
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
    dim3 gridDim(ceil(m/32), ceil(n/32));
    dim3 blockDim(32, 32);
    gemm_gpu_naive_kernel<<<gridDim,blockDim>>>(m,n,k,A,alpha,B,beta,C);
}