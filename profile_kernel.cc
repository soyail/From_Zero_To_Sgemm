#include <iostream>
#include <random>
#include <chrono>
#include <stdlib.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <functional>
#include <cublas_v2.h>

#include "gemm_gpu_cublas.h"
#include "gemm_gpu_naive.h"
#include "gemm_gpu_mem_coalesce.h"
#include "gemm_gpu_tiling.h"
// #include "gemm_gpu_1d_threadtiling.h"
#include "gemm_gpu_2d_threadtiling.h"
#include "gemm_gpu_vectorized_mem.h"
#include "gemm_gpu_bank_conflict.h"
#include "gemm_gpu_warptiling.h"
#include "gemm_gpu_doublebuffer.h"
#include "gemm_gpu_doublebuffer_sm2reg.h"



void run_kernel(int kernel_idx, int m, int n, int k, float* A, float alpha, float* B, float beta, float* C, cublasHandle_t handle){
    switch (kernel_idx)
    {
    case 0:
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B,
                n, A, k, &beta, C, n);
        break;
    case 1:
        gemm_gpu_naive(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 2:
        gemm_gpu_mem_coalesce(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 3:
        gemm_gpu_tiling(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 4:
        gemm_gpu_2d_threadtiling(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 5:
        gemm_gpu_vectorized_memory(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 6:
        gemm_gpu_bank_conflict(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 7:
        gemm_gpu_warptiling(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 8:
        gemm_gpu_doublebuffer_gm2sm(m,n,k,A,alpha,B,beta,C,handle);
        break;
    case 9:
        gemm_gpu_doublebuffer_sm2reg(m,n,k,A,alpha,B,beta,C,handle);
        break;
    
    default:
        break;
    }
}

int main(int argc, char* argv[]){
    const int m = 1024;
    const int n = 1024;
    const int k = 1024;
    int kernel_idx = atoi(argv[1]);
    // Allocate A,B,C on CPU
    float* A = new float[m*k];
    float* B = new float[k*n];
    float* C = new float[m*n];
    float* C_ref = new float[n*m];
    // Allocate A_gpu, B_gpu, C_gpu on GPU
    float* A_gpu, *B_gpu, *C_gpu;

    cudaMalloc(&A_gpu, m*k*sizeof(float));
    cudaMalloc(&B_gpu, k*n*sizeof(float));
    cudaMalloc(&C_gpu, n*m*sizeof(float));

    // initialize matrix.
    for (int i = 0; i < m * k; ++i) A[i] = static_cast<float>(rand())/RAND_MAX;
	for (int i = 0; i < k * n; ++i) B[i] = static_cast<float>(rand())/RAND_MAX;
	for (int i = 0; i < m * n; ++i) C[i] = 0.0;

    cudaMemcpy(A_gpu, A, m*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, k*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, m*n*sizeof(float), cudaMemcpyHostToDevice);

    // Create cublas handle
    cublasHandle_t handle;
    if(cublasCreate(&handle)){
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };
    float alpha = 1.0;
    float beta = 0.0;
    
    // run kernel
    run_kernel(kernel_idx,m,n,k,A_gpu,alpha,B_gpu,beta,C_gpu,handle);
    
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cublasDestroy(handle);
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
    return 0;
}


