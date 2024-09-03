#include <cublas_v2.h>

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
);


