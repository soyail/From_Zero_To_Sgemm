CXX = nvcc
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
CUDA_ARCH_FLAG := $(if $(CUDA_ARCH),-arch=sm_$(CUDA_ARCH),-arch=sm_70)
CXXFlAGS = --generate-line-info $(CUDA_ARCH_FLAG) -O3 -Iinclude
CXXFLAGS = $(CXXFlAGS)


SRC_DIR = src
APP_DIR = apps
INC_DIR = include

DEPS = $(wildcard $(INC_DIR)/*.h $(INC_DIR)/*.cuh)
SRC_CC = $(wildcard $(SRC_DIR)/*.cc)
SRC_CU = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cc,$(SRC_DIR)/%.o,$(SRC_CC)) \
	$(patsubst $(SRC_DIR)/%.cu,$(SRC_DIR)/%.o,$(SRC_CU))


$(SRC_DIR)/%.o: $(SRC_DIR)/%.cc $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@

$(APP_DIR)/%.o: $(APP_DIR)/%.cc $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@

$(APP_DIR)/%.o: $(APP_DIR)/%.cu $(DEPS)
	$(CXX) -c $(CXXFlAGS) $< -o $@


gemm_test: $(OBJS) $(APP_DIR)/gemm_test.o
	$(CXX) $(CXXFLAGS) -lcublas $^ -o gemm_test

bench_gemm: $(OBJS) $(APP_DIR)/bench_gemm.o
	$(CXX) $(CXXFLAGS) -lcublas $^ -o bench_gemm

profile_kernel: $(OBJS) $(APP_DIR)/profile_kernel.o
	$(CXX) $(CXXFLAGS) -lcublas $^ -o profile_kernel

query_gpu_properties: $(APP_DIR)/query_gpu_properties.o
	$(CXX) $(CXXFLAGS) $^ -o query_gpu_properties

clean:
	rm -f $(SRC_DIR)/*.o $(APP_DIR)/*.o gemm_test bench_gemm profile_kernel query_gpu_properties
