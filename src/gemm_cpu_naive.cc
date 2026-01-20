#pragma once

#include "gemm_cpu_naive.h"

__attribute__((optimize("O1")))
void gemm_cpu_naive(
    float* __restrict__ C, 
    float* __restrict__ A, 
    float* __restrict__ B, 
    const int n, 
    const int m, 
    const int k){
    for(int l=0; l<k; ++l)
        for(int i=0; i<n; ++i)
            for(int j=0; j<m; ++j){
                C[i*m+j] += A[i*k+l] * B[l*m+j];
            }
}