#include "gemm_gpu_cublas.h"

void gemm_gpu_cublas(
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
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B,
                n, A, k, &beta, C, n);
}
