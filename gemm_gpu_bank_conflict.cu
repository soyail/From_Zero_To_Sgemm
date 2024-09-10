#pragma once

#include "gemm_gpu_bank_conflict.h"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_gpu_bank_conflict_kernel(
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

    // calculating the indices that this thread will load into SMEM
    // because we use float4 here.
    const uint innerRowA = threadIdx.x / (BK/4); 
    const uint innerColA = threadIdx.x % (BK/4); 
    const uint innerRowB = threadIdx.x / (BN/4); 
    const uint innerColB = threadIdx.x % (BN/4);

    int threadNumsPerBlock = BM * BN / (TM*TN);
    int strideA = threadNumsPerBlock/(BK/4);
    int strideB = threadNumsPerBlock/(BN/4);

    for(uint tile_idx = 0; tile_idx < k; tile_idx += BK){
        // shareA tile: BM * BK, every thread need to load BK *TM *TN/ BN = 8.
        for(uint loadoffset = 0; loadoffset < BM; loadoffset+=strideA){
            // float4 tmp =
            //     reinterpret_cast<float4 *>(&A[(innerRowA+loadoffset)*k + innerColA*4])[0];
            // shareA[(innerColA * 4 + 0) * BM + innerRowA+loadoffset] = tmp.x;
            // shareA[(innerColA * 4 + 1) * BM + innerRowA+loadoffset] = tmp.y;
            // shareA[(innerColA * 4 + 2) * BM + innerRowA+loadoffset] = tmp.z;
            // shareA[(innerColA * 4 + 3) * BM + innerRowA+loadoffset] = tmp.w;
            reinterpret_cast<float4 *>(&shareA[(innerRowA+loadoffset)*BK+innerColA*4])[0] = 
                reinterpret_cast<float4 *>(&A[(innerRowA+loadoffset)*k+innerColA*4])[0];
        }
        for(uint loadoffset = 0; loadoffset < BK; loadoffset+=strideB){
            reinterpret_cast<float4 *>(&shareB[(innerRowB+loadoffset)*BN + innerColB*4])[0] =
                reinterpret_cast<float4 *>(&B[(innerRowB+loadoffset)*n + innerColB * 4])[0];

        }
        
        __syncthreads();

        A += BK;
        B += BK*n; 

        for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
            // load into register.
            for(uint i=0; i<TM; i++){
                regA[i] = shareA[(threadRow+i)*BK+dotIdx];
            }
            for(uint i=0; i<TN; i+=4){
                reinterpret_cast<float4 *>(&regB[i])[0] = 
                    reinterpret_cast<float4 *>(&shareB[dotIdx*BN+threadCol+i])[0];
            }
            for(uint resIdxM=0; resIdxM < TM; resIdxM++){
                for(uint resIdxN=0; resIdxN<TN; resIdxN++){
                    threadResults[resIdxM*TN+resIdxN] += regA[resIdxM]*regB[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for(uint resIdxM=0; resIdxM<TM; ++resIdxM){
        for(uint resIdxN=0; resIdxN < TN; resIdxN+=4){
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow + resIdxM) * n + threadCol + resIdxN])[0];
            tmp.x = threadResults[resIdxM*TN + resIdxN];
            tmp.y = threadResults[resIdxM*TN + resIdxN+1];
            tmp.z = threadResults[resIdxM*TN + resIdxN+2];
            tmp.w = threadResults[resIdxM*TN + resIdxN+3];
            reinterpret_cast<float4 *>(
                &C[(threadRow + resIdxM) * m + threadCol + resIdxN])[0] = tmp;
        }
    }
   

}

void gemm_gpu_bank_conflict(
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
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TM = 8;
    const int TN = 4;
    // another param
    // const int BM = 64;
    // const int BN = 64;
    // const int BK = 4;
    // const int TM = 8;
    // const int TN = 8;
    // block num 
    dim3 grid_dim = dim3(ceil(m/BM), ceil(n/BN));
    dim3 block_dim = dim3(BN*BM/(TM*TN));
    gemm_gpu_bank_conflict_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}