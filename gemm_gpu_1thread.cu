#include "gemm_gpu_1thread.h"

__global__ void gemm_gpu_1thread_kernel(
    float* __restrict__ C, 
    float* __restrict__ A, 
    float* __restrict__ B, 
    const int n, 
    const int m, 
    const int k){
    for(int i=0; i<n; ++i){
        for(int j=0; j<m; ++j){
            float tmp = 0.0;
            for(int l=0; l<k; ++l){
                tmp += A[i*k+l] * B[l*m+j];
            }
            C[i*m+j] = tmp;
        }
    }
}

void gemm_gpu_1thread(
    float* __restrict__ C, 
    float* __restrict__ A, 
    float* __restrict__ B, 
    const int n, 
    const int m, 
    const int k
){
    gemm_gpu_1thread_kernel<<<1,1>>>(C,A,B,n,m,k);
}


