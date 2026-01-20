// Auto-generated aggregation unit for GEMM kernels
#define GEMM_GPU_1D_THREADTILING_IMPLEMENTATION
#include "gemm_gpu_1d_threadtiling.cuh"

#define GEMM_GPU_1THREAD_IMPLEMENTATION
#include "gemm_gpu_1thread.cuh"

#define GEMM_GPU_2D_THREADTILING_IMPLEMENTATION
#include "gemm_gpu_2d_threadtiling.cuh"

#define GEMM_GPU_BANK_CONFLICT_IMPLEMENTATION
#include "gemm_gpu_bank_conflict.cuh"

#define GEMM_GPU_CUBLAS_IMPLEMENTATION
#include "gemm_gpu_cublas.cuh"

#define GEMM_GPU_DOUBLEBUFFER_IMPLEMENTATION
#include "gemm_gpu_doublebuffer.cuh"

#define GEMM_GPU_DOUBLEBUFFER_SM2REG_IMPLEMENTATION
#include "gemm_gpu_doublebuffer_sm2reg.cuh"

#define GEMM_GPU_MEM_COALESCE_IMPLEMENTATION
#include "gemm_gpu_mem_coalesce.cuh"

#define GEMM_GPU_NAIVE_IMPLEMENTATION
#include "gemm_gpu_naive.cuh"

#define GEMM_GPU_TILING_IMPLEMENTATION
#include "gemm_gpu_tiling.cuh"

#define GEMM_GPU_VECTORIZED_MEM_IMPLEMENTATION
#include "gemm_gpu_vectorized_mem.cuh"

#define GEMM_GPU_WARPTILING_IMPLEMENTATION
#include "gemm_gpu_warptiling.cuh"
