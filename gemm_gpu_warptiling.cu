#pragma once


#include "gemm_gpu_warptiling.h"

const int WARPSIZE = 32;

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void gemm_gpu_warptiling_kernel(
    const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C
){
    // Each SM is partitioned into four SM sub partition. 
    const uint warpIdx = threadIdx.x / WARPSIZE;
    const uint warpRow = warpIdx / (BN/WN);
    const uint warpCol = warpIdx % (BN/WN);

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; 
    constexpr uint WSUBN = WN / WNITER;

    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    //share memory.
    __shared__ float shareA[BM*BK];
    __shared__ float shareB[BK*BN];
    
    // TODO: blockIdx.x -> blockIdx.y
    // one block calculate BM*BN elements.
    // There're NUM_THREADS / WARPSIZE warp in the block.
    A += blockIdx.x * BM * k;
    B += blockIdx.y * BN;
    C += (blockIdx.x * BM+ warpRow * WM) * n + blockIdx.y * BN + warpCol*WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // use register to store result.
    float threadResults[WMITER*TM*WNITER*TN] = {0.0};
    float regA[WMITER*TM] = {0.0};
    float regB[WNITER*TN] = {0.0};
    
    for(uint tile_idx = 0; tile_idx < k; tile_idx += BK){
        for(uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA){
            float4 tmp =
                reinterpret_cast<float4 *>(&A[(innerRowA + offset) * k + innerColA * 4])[0];
            shareA[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            shareA[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            shareA[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            shareA[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4 *>(&shareB[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(&B[(innerRowB + offset) * n + innerColB * 4])[0];
        }
        
        __syncthreads();

        A += BK;
        B += BK*n; 

        for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
            for(uint wSubRowIdx=0; wSubRowIdx<WMITER; ++wSubRowIdx){
                for(uint i=0; i<TM; i++){
                    regA[wSubRowIdx*TM+i] = shareA[dotIdx*BM+warpRow*WM+wSubRowIdx*WSUBM+threadRowInWarp*TM+i];
                }
            }
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i = 0; i < TN; ++i) {
                    regB[wSubColIdx * TN + i] =
                        shareB[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                        threadColInWarp * TN + i];
                }
            }
            
            
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    // calculate per-thread results
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                        (wSubColIdx * TN) + resIdxN] +=
                                regA[wSubRowIdx * TM + resIdxM] *
                                regB[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // move C pointer to current warp subtile
            float *C_interim = C + (wSubRowIdx * WSUBM) * n + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                // load C vector into registers
                float4 tmp = reinterpret_cast<float4 *>(
                    &C_interim[(threadRowInWarp * TM + resIdxM) * n +
                                threadColInWarp * TN + resIdxN])[0];
                // perform GEMM update in reg
                const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                wSubColIdx * TN + resIdxN;
                tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                // write back
                reinterpret_cast<float4 *>(
                    &C_interim[(threadRowInWarp * TM + resIdxM) * n +
                                threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
   

}

void gemm_gpu_warptiling(
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
    const int WM = 32;
    const int WN = 32;
    const int WNITER = 2;
    const int TM = 4;
    const int TN = 4;
    const int NUM_THREADS = 128;

    dim3 grid_dim = dim3(ceil(m/BM), ceil(n/BN));
    dim3 block_dim = dim3(NUM_THREADS);
    gemm_gpu_warptiling_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}