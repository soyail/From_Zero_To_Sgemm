#pragma once

#include <cassert>
#include "gemm_gpu_tiling.h"


template <const int BM, const int BN, const int BK>
__global__ void gemm_gpu_tiling_kernel(
    const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C
){
    
    __shared__ float shareA[BM][BK];
    __shared__ float shareB[BK][BN];
    
    A += blockIdx.x * BM * k;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * n + blockIdx.y * BN;
    
    float tmp = 0.0;
    for(int i=0; i<k/BK;++i){
        shareA[threadIdx.y][threadIdx.x] = A[threadIdx.y*k+threadIdx.x];
        shareB[threadIdx.y][threadIdx.x] = B[threadIdx.y*n+threadIdx.x];
        A += BK;
        B += BK * n;
        __syncthreads();
        for(int j=0; j<BK; ++j){
            tmp += shareA[threadIdx.y][j] * shareB[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[threadIdx.y*n+threadIdx.x] = alpha*tmp+beta*C[threadIdx.y*n+threadIdx.x];
}

void gemm_gpu_tiling(
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
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    dim3 grid_dim = dim3(ceil(m/BM), ceil(n/BN));
    dim3 block_dim = dim3(BM, BN);
    gemm_gpu_tiling_kernel<BM,BN,BK><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}