#pragma once

#include <cassert>
#include "gemm_gpu_2d_threadtiling.h"


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_gpu_2d_threadtiling_kernel(
    const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C
){
    //share memory.
    __shared__ float shareA[BM*BK];
    __shared__ float shareB[BK*BN];

    // use register to store result.
    float threadResults[TM*TN] = {0.0};
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};
   
   
    // calculating the indices that this thread will be responsible for the result C.
    const int threadRow = (threadIdx.x*TN / BN)*TM;
    const int threadCol = threadIdx.x*TN % BN;

    A += blockIdx.x * BM * k;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * n + blockIdx.y * BN;

    int threadNumsPerBlock = BM * BN / (TM*TN);
    int strideA = threadNumsPerBlock/BK;
    int strideB = threadNumsPerBlock/BN;

    // calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK; // [0,1,2,...,BM]
    const uint innerColA = threadIdx.x % BK; 
    const uint innerRowB = threadIdx.x / BN; // [0,1,2,...,BK]
    const uint innerColB = threadIdx.x % BN;

    for(uint tile_idx = 0; tile_idx < k; tile_idx += BK){
        for(uint loadoffset = 0; loadoffset < BM; loadoffset+=strideA){
            shareA[(innerRowA+loadoffset)*BK+innerColA] = A[(innerRowA+loadoffset)*k+innerColA]; 
        }
        for(uint loadoffset = 0; loadoffset < BK; loadoffset+=strideB){
            shareB[(innerRowB+loadoffset)*BN+innerColB] = B[(innerRowB+loadoffset)*n+innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK*n; 
        
        for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
            // load into register.
            for(uint i=0; i<TM; i++){
                regA[i] = shareA[(threadRow+i)*BK+dotIdx];
            }
            for(uint i=0; i<TN; i++){
                regB[i] = shareB[dotIdx*BN+threadCol+i];
            }
            for(uint resIdxM=0; resIdxM < TM; resIdxM++){
                for(uint resIdxN=0; resIdxN<TN; resIdxN++){
                    threadResults[resIdxM*TN+resIdxN] += regA[resIdxM]*regB[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    for(uint resIdxM=0; resIdxM < TM; resIdxM++){
        for(uint resIdxN=0; resIdxN<TN; resIdxN++){
            C[(threadRow+resIdxM)*n+threadCol+resIdxN] = threadResults[resIdxM*TN+resIdxN];
            // float4 tmp = reinterpret_cast<float4 *>(
            //     &C[(threadRow + resIdxM) * n + threadCol + resIdxN])[0];
            // tmp.x = threadResults[resIdxM*TN + resIdxN];
            // tmp.y = threadResults[resIdxM*TN + resIdxN+1];
            // tmp.z = threadResults[resIdxM*TN + resIdxN+2];
            // tmp.w = threadResults[resIdxM*TN + resIdxN+3];
            // reinterpret_cast<float4 *>(
            //     &C[(threadRow + resIdxM) * m + threadCol + resIdxN])[0] = tmp;
        }
    }

}

void gemm_gpu_2d_threadtiling(
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
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 4;
    // const int BM = 64;
    // const int BN = 64;
    // const int BK = 16;
    // const int TM = 8;
    // const int TN = 4;

    dim3 grid_dim = dim3(ceil(m/BM), ceil(n/BN));
    dim3 block_dim = dim3(BN*BM/(TM*TN));
    gemm_gpu_2d_threadtiling_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}