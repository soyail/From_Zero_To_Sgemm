#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemm_gpu_bank_conflict.cuh"
#include "gemm_gpu_cublas.cuh"
#include "gemm_gpu_mem_coalesce.cuh"
#include "gemm_gpu_naive.cuh"
#include "gemm_gpu_tiling.cuh"
#include "gemm_gpu_vectorized_mem.cuh"
#include "gemm_gpu_warptiling.cuh"
#include "gemm_gpu_2d_threadtiling.cuh"

typedef void (*gemm_impl_t)(
    const int m,
    const int n,
    const int k,
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C,
    cublasHandle_t handle);

struct GemmImpl {
    std::string name;
    gemm_impl_t impl;
};

std::vector<GemmImpl> gemm_impls = {
    {"gpu_naive", gemm_gpu_naive},
    {"gpu_cublas", gemm_gpu_cublas},
    {"gpu_mem_coalesce", gemm_gpu_mem_coalesce},
    {"gpu_tiling", gemm_gpu_tiling},
    {"gpu_2d_threadtiling", gemm_gpu_2d_threadtiling},
    {"gpu_vectorized_memory", gemm_gpu_vectorized_memory},
    {"gpu_bank_conflict", gemm_gpu_bank_conflict},
    {"gpu_warptiling", gemm_gpu_warptiling},
};

static std::vector<int> parse_k_list(const std::string &list) {
    std::vector<int> ks;
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        int k = std::atoi(item.c_str());
        if (k > 0) ks.push_back(k);
    }
    return ks;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::fprintf(
            stderr,
            "Usage: %s m n k_list [impl_name] [rounds] [out_csv]\n",
            argv[0]);
        return 1;
    }
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    std::string k_list = argv[3];
    assert(m > 0 && n > 0);
    std::vector<int> ks = parse_k_list(k_list);
    if (ks.empty()) {
        std::fprintf(stderr, "k_list is empty or invalid.\n");
        return 1;
    }
    std::string impl_filter;
    if (argc >= 5) {
        impl_filter = argv[4];
    }
    int rounds = 50;
    if (argc >= 6) {
        rounds = std::atoi(argv[5]);
        if (rounds <= 0) rounds = 50;
    }
    std::string out_csv = "benchmark.csv";
    if (argc >= 7) {
        out_csv = argv[6];
    }

    float *A = new float[m * k];
    float *B = new float[k * n];
    float *C = new float[m * n];
    float *A_gpu = nullptr;
    float *B_gpu = nullptr;
    float *C_gpu = nullptr;

    cudaMalloc(&A_gpu, m * k * sizeof(float));
    cudaMalloc(&B_gpu, k * n * sizeof(float));
    cudaMalloc(&C_gpu, m * n * sizeof(float));

    for (int i = 0; i < m * k; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < m * n; ++i) C[i] = 0.0f;

    cudaMemcpy(A_gpu, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::fprintf(stderr, "Create cublas handle error.\n");
        return 1;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    std::ofstream csv(out_csv);
    if (!csv) {
        std::fprintf(stderr, "Failed to open output file: %s\n", out_csv.c_str());
        return 1;
    }
    csv << "impl,m,n,k,ms,gflops\n";
    for (int k : ks) {
        for (auto gemm_impl : gemm_impls) {
            if (!impl_filter.empty() &&
                gemm_impl.name != impl_filter &&
                gemm_impl.name != "gpu_cublas") {
                continue;
            }

            cudaMemset(C_gpu, 0, m * n * sizeof(float));
            gemm_impl.impl(m, n, k, A_gpu, alpha, B_gpu, beta, C_gpu, handle);
            cudaDeviceSynchronize();

            cudaEventRecord(start, 0);
            for (int round = 0; round < rounds; ++round) {
                cudaMemset(C_gpu, 0, m * n * sizeof(float));
                gemm_impl.impl(m, n, k, A_gpu, alpha, B_gpu, beta, C_gpu, handle);
            }
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);

            float elapsed_ms = 0.0f;
            cudaEventElapsedTime(&elapsed_ms, start, end);
            long flops = 2L * m * n * k;
            float avg_ms = elapsed_ms / rounds;
            float gflops = (rounds * flops * 1e-9f) / elapsed_ms;
            csv << gemm_impl.name << "," << m << "," << n << "," << k << ","
                << avg_ms << "," << gflops << "\n";
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
