#pragma once


#include "gemm_gpu_doublebuffer.h"


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_gpu_doublebuffer_kernel(
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
    // avoid bank conflict.
    __shared__ float shareA[2][BK][BM+1];
    __shared__ float shareB[2*BK*BN];

    // use register to store result.
    float threadResults[TM*TN] = {0.0};
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    // move pointer.
    A += blockIdx.x * BM * k;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * n + blockIdx.y * BN;


    // calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / (BK/4); 
    const uint innerColA = threadIdx.x % (BK/4); 
    const uint innerRowB = threadIdx.x / (BN/4); 
    const uint innerColB = threadIdx.x % (BN/4);

    // calculating the indices that this thread will be responsible for the result C.
    const int threadRow = (threadIdx.x*TN / BN)*TM;
    const int threadCol = threadIdx.x*TN % BN;

    int threadNumsPerBlock = BM * BN / (TM*TN);
    int strideA = threadNumsPerBlock/(BK/4);
    int strideB = threadNumsPerBlock/(BN/4);

    // load the first tile of shareA,shareB from GMEM to SMEM.
    for(uint loadoffset = 0; loadoffset < BM; loadoffset+=strideA){
        // transpose shareA.
        float4 tmp =
            reinterpret_cast<float4 *>(&A[(innerRowA+loadoffset)*k + innerColA*4])[0];
        shareA[0][(innerColA * 4 + 0)][innerRowA+loadoffset] = tmp.x;
        shareA[0][(innerColA * 4 + 1)][innerRowA+loadoffset] = tmp.y;
        shareA[0][(innerColA * 4 + 2)][innerRowA+loadoffset] = tmp.z;
        shareA[0][(innerColA * 4 + 3)][innerRowA+loadoffset] = tmp.w;
    }
    for(uint loadoffset = 0; loadoffset < BK; loadoffset+=strideB){
        reinterpret_cast<float4 *>(&shareB[(innerRowB+loadoffset)*BN + innerColB*4])[0] =
            reinterpret_cast<float4 *>(&B[(innerRowB+loadoffset)*n + innerColB*4])[0];
    }
        
    // must wait all threads complete.
    __syncthreads();

    int offset = 1;

    A += BK;
    B += BK*n; 

    for(uint tile_idx = BK; tile_idx < k; tile_idx += BK){
        // load next tile from global memory(GMEM to temp register).
        for(uint loadoffset = 0; loadoffset < BM; loadoffset+=strideA){
            float4 tmp =
                reinterpret_cast<float4 *>(&A[(innerRowA+loadoffset)*k + innerColA*4])[0];
            shareA[offset][(innerColA * 4 + 0)][innerRowA+loadoffset] = tmp.x;
            shareA[offset][(innerColA * 4 + 1)][innerRowA+loadoffset] = tmp.y;
            shareA[offset][(innerColA * 4 + 2)][innerRowA+loadoffset] = tmp.z;
            shareA[offset][(innerColA * 4 + 3)][innerRowA+loadoffset] = tmp.w;
        }
        for(uint loadoffset = 0; loadoffset < BK; loadoffset+=strideB){
            reinterpret_cast<float4 *>(&shareB[offset*BN*BK+(innerRowB+loadoffset)*BN + innerColB*4])[0] =
                reinterpret_cast<float4 *>(&B[(innerRowB+loadoffset)*n + innerColB * 4])[0];
        }

        A += BK;
        B += BK*m; 

        offset = !offset;
        // process current tile.
        for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
            // load into register.
            for(uint i=0; i<TM; i++){
                // regA[i] = shareA[offset*BM*BK+(threadRow+i)*BK+dotIdx];
                reinterpret_cast<float4 *>(&regA[i])[0] = 
                    reinterpret_cast<float4 *>(&shareA[offset][dotIdx][threadRow+i])[0];
            }
            for(uint i=0; i<TN; i+=4){
                reinterpret_cast<float4 *>(&regB[i])[0] = 
                    reinterpret_cast<float4 *>(&shareB[offset*BK*BN+dotIdx*BN+threadCol+i])[0];
            }
            for(uint resIdxM=0; resIdxM < TM; resIdxM++){
                for(uint resIdxN=0; resIdxN<TN; resIdxN++){
                    threadResults[resIdxM*TN+resIdxN] += regA[resIdxM]*regB[resIdxN];
                }
            }
        }
        __syncthreads();
        
    }
    // process the final loading data.
    offset = !offset;
    for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
        // load into register.
        for(uint i=0; i<TM; i++){
            // regA[i] = shareA[offset*BM*BK+(threadRow+i)*BK+dotIdx];
            reinterpret_cast<float4 *>(&regA[i])[0] = 
                reinterpret_cast<float4 *>(&shareA[offset][dotIdx][threadRow+i])[0];
        }
        for(uint i=0; i<TN; i++){
            reinterpret_cast<float4 *>(&regB[i])[0] = 
                reinterpret_cast<float4 *>(&shareB[offset*BK*BN+dotIdx*BN+threadCol+i])[0];
        }
        for(uint resIdxM=0; resIdxM < TM; resIdxM++){
            for(uint resIdxN=0; resIdxN<TN; resIdxN++){
                threadResults[resIdxM*TN+resIdxN] += regA[resIdxM]*regB[resIdxN];
            }
        }
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

void gemm_gpu_doublebuffer_gm2sm(
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

    dim3 grid_dim = dim3(ceil(m/BM), ceil(n/BN));
    dim3 block_dim = dim3(BN*BM/(TM*TN));
    gemm_gpu_doublebuffer_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}