#include <cublas_v2.h>

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
);