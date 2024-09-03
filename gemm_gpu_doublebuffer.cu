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
    __shared__ float shareA[2*BM*BK];
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

    // load shareA,shareB from GMEM to SMEM.
    float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * k + innerColA * 4])[0];
    shareA[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    shareA[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    shareA[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    shareA[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
    reinterpret_cast<float4 *>(&shareB[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * n + innerColB * 4])[0];

    // must wait all threads complete.
    __syncthreads();

    int write_matrix_idx = 1;

    A += BK;
    B += BK*m; 

    for(uint tile_idx = BK; tile_idx < k; tile_idx += BK){
        // load next tile from global memory(GMEM to temp register).
        float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * k + innerColA * 4])[0];
        shareA[write_matrix_idx*BM*BK+(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        shareA[write_matrix_idx*BM*BK+(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        shareA[write_matrix_idx*BM*BK+(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        shareA[write_matrix_idx*BM*BK+(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
        reinterpret_cast<float4 *>(&shareB[write_matrix_idx*BK*BN+innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * n + innerColB * 4])[0];

        A += BK;
        B += BK*m; 

        write_matrix_idx = !write_matrix_idx;
        // process current tile.
        for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
            // load into register.
            for(uint i=0; i<TM; i++){
                regA[i] = shareA[write_matrix_idx*BM*BK+dotIdx*BM+threadRow+i];
            }
            for(uint i=0; i<TN; i++){
                regB[i] = shareB[write_matrix_idx*BK*BN+dotIdx*BM+threadCol+i];
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
    write_matrix_idx = !write_matrix_idx;
    for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
        // load into register.
        for(uint i=0; i<TM; i++){
            regA[i] = shareA[write_matrix_idx*BM*BK+dotIdx*BM+threadRow+i];
        }
        for(uint i=0; i<TN; i++){
            regB[i] = shareB[write_matrix_idx*BK*BN+dotIdx*BM+threadCol+i];
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
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    dim3 grid_dim = dim3(ceil(m/BM), ceil(n/BN));
    dim3 block_dim = dim3(BN*BM/(TM*TN));
    gemm_gpu_doublebuffer_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}

