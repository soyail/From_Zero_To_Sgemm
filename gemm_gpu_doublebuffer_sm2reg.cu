#include "gemm_gpu_doublebuffer_sm2reg.h"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_gpu_doublebuffer_sm2reg_kernel(
    const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C
){
    // allocate shared memory.
    __shared__ float shareA[BM*BK];
    __shared__ float shareB[BK*BN];
    // 
    float regA[2*TM] = {0.0};
    float regB[2*TN] = {0.0};
    float res[TM*TN] = {0.0};

    int blockRow = blockIdx.x * BM;
    int blockCol = blockIdx.y * BN; 

    // move pointer;
    A += blockRow*k;
    B += blockCol;
    C += blockRow * n + blockCol;

    // in this case, every thread need to load 4 elements from A and B.
    // calculate the indice of shareA that this thread will be responsible for loading.
    int inner_row_A = threadIdx.x * 4 / BK;
    int inner_col_A = threadIdx.x * 4 % BK;
    int inner_row_B = threadIdx.x * 4 / BN;
    int inner_col_B = threadIdx.x * 4 % BN;
    // calculate the indice of matrix C that this thread will be responsible for.
    int threadRow = (threadIdx.x*TN  / BN)*TM;
    int threadCol = threadIdx.x*TN % BN;

    // split on dimension k by step BK;
    for(int tile_k=0; tile_k+=BK; tile_k<k){
        // load from GMEM to SMEM;
        // transpose.
        float4 tmp =
            reinterpret_cast<float4 *>(&A[inner_row_A * k + inner_col_A])[0];
        shareA[(inner_col_A * 4 + 0) * BM + inner_row_A] = tmp.x;
        shareA[(inner_col_A * 4 + 1) * BM + inner_row_A] = tmp.y;
        shareA[(inner_col_A * 4 + 2) * BM + inner_row_A] = tmp.z;
        shareA[(inner_col_A * 4 + 3) * BM + inner_row_A] = tmp.w;

        reinterpret_cast<float4 *>(&shareB[threadIdx.x*4])[0] = 
            reinterpret_cast<float4 *>(&B[inner_row_B*n+inner_col_B])[0];

        // load the first tile of shareA to reg.
        for(int resIdxM=0; resIdxM<TM; resIdxM++){
            regA[resIdxM] = shareA[threadRow+resIdxM];
        }
        for(int resIdxN=0; resIdxN<TN; resIdxN++){
            regB[resIdxN] = shareB[threadCol+resIdxN];
        }
        __syncthreads();
        // move pointer;
        A += BK;
        B += BK*n;

        int cur_tile = 1;

        for(int dotIdx=0; dotIdx<BK-1; ++dotIdx){
            // threads will access share memory meanwhile
            // which will cause bank conflict.
            for(int resIdxM=0; resIdxM<TM; resIdxM++){
                regA[cur_tile*BM*BK+resIdxM] = shareA[(dotIdx+1)*BK+threadRow+resIdxM];
            }
            for(int resIdxN=0; resIdxN<TN; resIdxN++){
                regB[cur_tile*BK*BN+resIdxN] = shareB[(dotIdx+1)*BK+threadCol+resIdxN]; 
            }
            // switch to another tile while processing data; 
            cur_tile = !cur_tile;
            for(int resIdxM=0; resIdxM<TM; resIdxM++){
                for(int resIdxN=0; resIdxN<TN; resIdxN++){
                    res[resIdxM*TN+resIdxN] = regA[cur_tile*BM*BK+resIdxM]*regB[cur_tile*BK*BN+resIdxN];
                }
            }
        }
        // process the last col of BK;
        cur_tile = !cur_tile;
        for(int resIdxM=0; resIdxM<TM; resIdxM++){
            for(int resIdxN=0; resIdxN<TN; resIdxN++){
                res[resIdxM*TN+resIdxN] = regA[cur_tile*BM*BK+resIdxM]*regB[cur_tile*BK*BN+resIdxN];
            }
        }
        __syncthreads();
    }
    // copy results.
    for(uint resIdxM=0; resIdxM<TM; ++resIdxM){
        for(uint resIdxN=0; resIdxN < TN; resIdxN+=4){
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow + resIdxM) * n + threadCol + resIdxN])[0];
            tmp.x = res[resIdxM*TN + resIdxN];
            tmp.y = res[resIdxM*TN + resIdxN+1];
            tmp.z = res[resIdxM*TN + resIdxN+2];
            tmp.w = res[resIdxM*TN + resIdxN+3];
            reinterpret_cast<float4 *>(
                &C[(threadRow + resIdxM) * m + threadCol + resIdxN])[0] = tmp;
        }
    }
}

void gemm_gpu_doublebuffer_sm2reg(
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
    gemm_gpu_doublebuffer_sm2reg_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(m,n,k,A,alpha,B,beta,C);
}