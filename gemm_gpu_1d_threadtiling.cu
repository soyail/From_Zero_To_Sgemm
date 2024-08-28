#pragma once

#include <cassert>
#include "gemm_gpu_1d_threadtiling.h"


template <const int BM, const int BN, const int BK, const int TM>
__global__ void gemm_gpu_1d_threadtiling_kernel(
    const int m, 
    const int n, 
    const int k,
    const float *A,
    float alpha,
    const float *B,
    float beta,
    float *C
){
    //share memory.
    __shared__ float shareA[BM][BK];
    __shared__ float shareB[BK][BN];

    // register data.
    float threadResults[TM] = {0.0};

    // total BN * BM / TM thread.
    // Batch A : [BN, BK]
    // Batch B : [BM, BK] 
    int innerColA = threadIdx.x % BK;
    int innerRowA = threadIdx.x / BK;
    int innerColB = threadIdx.x % BN;
    int innerRowB = threadIdx.x / BN;
    int cCol = threadIdx.x % BN + blockIdx.y * BN;
    int cRow = threadIdx.x / BN * TM + blockIdx.x * BM;

    for(int i=0; i<k; i+=BK){
        shareA[innerRowA][innerColA] = A[(blockIdx.x * BM + innerRowA) * k + (i + innerColA)];
        shareB[innerRowB][innerColB] = B[(i+innerRowB)*n + blockIdx.y * BN + innerColB];
        __syncthreads();
        for(int j=0; j<BK; ++j){
            float tmp_b = shareB[j][innerColB];
            for(int resIdx=0; resIdx<TM; ++resIdx){
                threadResults[resIdx] += shareA[(threadIdx.x / BN * TM + resIdx)][j] * tmp_b;
            //tmp += shareA[threadIdx.y][j] * shareB[j][threadIdx.x];
            }
        }
        __syncthreads();
    }
    for(int resIdx=0; resIdx<TM; ++resIdx){
        C[(threadIdx.x/BN*TM+resIdx + blockIdx.x * BM) * n  + blockIdx.y * BN + threadIdx.x%BN] = threadResults[resIdx];
    }

}

void gemm_gpu_1d_threadtiling(
    const int m, 
    const int n, 
    const int k,
    const float *A,
    float alpha,
    const float *B,
    float beta,
    float *C,
    cublasHandle_t handle
){
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    assert(BM == BN);
    assert(BM == TM * BK);

    dim3 grid_dim = dim3(ceil(n/BN), ceil(m/BM));
    dim3 block_dim = dim3(BM*BN/TM);
    gemm_gpu_1d_threadtiling_kernel<BM, BN, BK, TM><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}