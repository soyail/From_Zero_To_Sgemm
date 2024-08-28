#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, dev);

        if (err != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: (" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate << " kHz" << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem << " bytes" << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Multi-processor count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;
        std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 cache size: " << deviceProp.l2CacheSize << " bytes" << std::endl;
        std::cout << "  Maximum texture dimension size (1D): " << deviceProp.maxTexture1D << std::endl;
        std::cout << "  Maximum texture dimension size (2D): (" << deviceProp.maxTexture2D[0] << ", "
                  << deviceProp.maxTexture2D[1] << ")" << std::endl;
        std::cout << "  Maximum texture dimension size (3D): (" << deviceProp.maxTexture3D[0] << ", "
                  << deviceProp.maxTexture3D[1] << ", " << deviceProp.maxTexture3D[2] << ")" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
