CXX = nvcc
CXXFlAGS = --generate-line-info -arch=sm_89 -O3


DEPS = gemm_gpu_cublas.h gemm_gpu_naive.h gemm_gpu_tiling.h gemm_gpu_mem_coalesce.h gemm_gpu_1d_threadtiling.h gemm_gpu_2d_threadtiling.h gemm_gpu_vectorized_mem.h gemm_gpu_bank_conflict.h gemm_gpu_warptiling.h gemm_gpu_doublebuffer.h gemm_gpu_doublebuffer_sm2reg.h
OBJS = gemm_gpu_cublas.o gemm_gpu_naive.o gemm_gpu_tiling.o gemm_gpu_mem_coalesce.o gemm_gpu_1d_threadtiling.o gemm_gpu_2d_threadtiling.o gemm_gpu_vectorized_mem.o gemm_gpu_bank_conflict.o gemm_gpu_warptiling.o gemm_gpu_doublebuffer.o gemm_gpu_doublebuffer_sm2reg.o 


%.o: %.cc $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@

%.o: %.cu $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@


gemm_test: $(OBJS) gemm_test.o
	$(CXX) $(CXXFLAGS) -lcublas $^ -o gemm_test

profile_kernel: $(OBJS) profile_kernel.o
	$(CXX) $(CXXFLAGS) -arch=sm_89 -lcublas $^ -o profile_kernel

clean:
	rm -f *.o gemm_test profile_kernel