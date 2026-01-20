# From_Zero_To_Sgemm

This project implements and optimizes several CUDA SGEMM kernels and compares
their performance against cuBLAS.

## Layout
- `include/`: CUDA kernel declarations and implementations in `.cuh`
- `src/`: kernel compilation units and CPU helpers
- `apps/`: runnable executables (tests, benchmarks, utilities)
- `benchmark.py`: plot performance from CSV output

## Build
All executables are built with `nvcc`:
```sh
make gemm_test
make bench_gemm
make profile_kernel
make query_gpu_properties
```

## Run
### Functional + performance test
```sh
./gemm_test m n k
```

### Benchmark sweep (CSV output)
```sh
./bench_gemm 4096 4096 256,512,1024,2048,4096 gpu_tiling 50 benchmark.csv
```

### Plot from CSV
```sh
python3 benchmark.py --impl gpu_tiling --csv benchmark.csv
```

### Query GPU properties
```sh
./query_gpu_properties
```

## Notes
- `bench_gemm` benchmarks the requested implementation plus `gpu_cublas`.
- If `nvidia-smi` is available, `make` picks the GPU compute capability
  automatically; otherwise it falls back to `sm_70`.
