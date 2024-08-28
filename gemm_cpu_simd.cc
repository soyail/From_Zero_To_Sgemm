#include "gemm_cpu_simd.h"


void gemm_cpu_simd(
    float* __restrict__ C, 
    float* __restrict__ A, 
    float* __restrict__ B, 
    const int n, 
    const int m, 
    const int k){
    // The first edition slower than the second edition.
    /*
    for(int i=0; i<n; ++i){
        for(int j=0; j<m; ++j){
            for(int l=0; l<k; ++l){
                C[i*m+j] += A[i*k+l] * B[l*m+j];
            }
        }
    }
    */
    
    // The Second edition
   for(int l=0; l<k; ++l){
        for(int i=0; i<n; ++i){
            for(int j=0; j<m; ++j){
                C[i*m+j] += A[i*k+l] * B[l*m+j];
            }
        }
    }
   
    
}