#pragma once

void gemm_cpu_simd(
    float* __restrict__ C, // [n,m]
    float* __restrict__ A, // [n,k]
    float* __restrict__ B, // [k,m]
    const int n,
    const int k,
    const int m
);