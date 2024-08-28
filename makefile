CXX = nvcc
CXXFlAGS = -O3


DEPS = gemm_gpu_cublas.h gemm_gpu_naive.h gemm_gpu_tiling.h gemm_gpu_mem_coalesce.h gemm_gpu_1d_threadtiling.h gemm_gpu_2d_threadtiling.h gemm_gpu_vectorized_mem.h gemm_gpu_doublebuffer.h
OBJS = gemm_gpu_cublas.o gemm_gpu_naive.o gemm_gpu_tiling.o gemm_gpu_mem_coalesce.o gemm_gpu_1d_threadtiling.o gemm_gpu_2d_threadtiling.o gemm_gpu_vectorized_mem.o gemm_gpu_doublebuffer.o gemm_test.o


%.o: %.cc $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@

%.o: %.cu $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@


gemm_test: $(OBJS)
	$(CXX) $(CXXFLAGS) -lcublas $^ -o gemm_test

clean:
	rm -f *.o gemm_test