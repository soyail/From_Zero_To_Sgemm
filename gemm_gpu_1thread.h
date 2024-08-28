

void gemm_gpu_1thread(
    float* __restrict__ C, 
    float* __restrict__ A, 
    float* __restrict__ B, 
    const int n, 
    const int m, 
    const int k
);