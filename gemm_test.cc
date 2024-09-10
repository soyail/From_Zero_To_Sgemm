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

typedef void (*gemm_impl_t)(
	const int m, 
    const int n, 
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C,
    cublasHandle_t handle
);

struct GemmImpl {
	std::string name;
	gemm_impl_t impl;
};

std::vector<GemmImpl> gemm_impls = {
    // {"cpu_naive", gemm_cpu_naive, false},
    {"gpu_naive", gemm_gpu_naive},
    {"gpu_cublas", gemm_gpu_cublas},
    {"gpu_mem_coalesce", gemm_gpu_mem_coalesce},
    {"gpu_tiling", gemm_gpu_tiling},
    // {"gpu_1d_threadtiling", gemm_gpu_1d_threadtiling},
    {"gpu_2d_threadtiling", gemm_gpu_2d_threadtiling},
    {"gpu_vectorized_memory", gemm_gpu_vectorized_memory},
    {"gpu_bank_conflict", gemm_gpu_bank_conflict},
    {"gpu_warptiling", gemm_gpu_warptiling},
    {"gpu_doublebuffer_gm2sm", gemm_gpu_doublebuffer_gm2sm}
    // {"gpu_doublebuffer_sm2reg", gemm_gpu_doublebuffer_sm2reg}
};

constexpr int BENCHMARK_ROUNDS = 50;

int main(int argc, char* argv[]){
    int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	assert (n > 0 && m > 0 && k > 0);
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
    // C = alpha*A*B + beta*C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_gpu,
                n, A_gpu, k, &beta, C_gpu, n);
    // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_gpu, CUDA_R_32F,
    //             n, A_gpu,CUDA_R_32F, k, &beta, C_gpu, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaMemcpy(C_ref, C_gpu, m*n*sizeof(float), cudaMemcpyDeviceToHost);

    float elapsed_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    
    // single precision matrix multiply
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for(auto gemm_impl:gemm_impls){
        // Verify Correctness
        cudaMemset(C_gpu, 0.0, m*n*sizeof(float));
        gemm_impl.impl(m,n,k,A_gpu,alpha,B_gpu,beta,C_gpu,handle);
        cudaMemcpy(C, C_gpu, m*n*sizeof(float), cudaMemcpyDeviceToHost);
        for(int idx=0; idx<m*n; idx++){
            if(std::fabs(C[idx]-C_ref[idx])>1e-2){
                printf("Verify Failed!\n");
                printf("idx: %d, C: %f, C_ref: %f\n",idx, C[idx], C_ref[idx]);
                break;
            }
        }

        cudaEventRecord(start,0);
        for(int round=0; round<BENCHMARK_ROUNDS; round++){
            cudaMemset(C_gpu, 0.0, m*n*sizeof(float));
            gemm_impl.impl(m,n,k,A_gpu,alpha,B_gpu,beta,C_gpu,handle);
        }
        cudaEventRecord(end,0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, start, end);
        long flops = 2*m*n*k;
        printf(
            "(%s): \n"
            "Average elapsed time: (%7.6f) ms, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            gemm_impl.name.c_str(),
            elapsed_time / BENCHMARK_ROUNDS,
            (BENCHMARK_ROUNDS * flops * 1e-9) / elapsed_time, m);
    }
    
    
    //gemm_cpu_naive(C_ref, A, B, n, m, k);

    // std::vector<std::pair<std::string, double>> results;
    // for(auto gemm_impl : gemm_impls){
    //     printf("Benchmarking function %s ...\n", gemm_impl.name.c_str());
    //     double avg_round_time_usage = benchmark_gemm_impl(gemm_impl,n,m,k);
    //     results.push_back({gemm_impl.name, avg_round_time_usage});
    // }
    // printf("----------------------\n");
    // printf("Results:\n");
    // for(auto result : results){
    //     printf("    %s      %lf us\n",result.first.c_str(), result.second);
    // }
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