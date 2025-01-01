#include <iostream>
#include <cuda_runtime.h>

void printGpuInfo() {
    int deviceCount = 0;

    // 获取 GPU 数量
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return;
    }

    // 获取每个 GPU 的信息
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
        std::cout << "  Memory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;
        std::cout << "  Total Global Memory (bytes): " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA Cores: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    }
}

// 计算 CUDA 核心数
int _ConvertSMVer2Cores(int major, int minor) {
    // NVIDIA GPU架构的每个版本都有不同的核心数
    // 通过计算每个版本的核心数来得出
    switch (major) {
        case 2: return (minor == 1) ? 48 : 32;
        case 3: return (minor == 0) ? 192 : (minor == 5) ? 128 : 192;
        case 5: return (minor == 0) ? 128 : 64;
        case 6: return (minor == 0) ? 128 : 64;
        case 7: return (minor == 0) ? 64 : 128;
        case 8: return (minor == 0) ? 128 : 64;
        default: return 0;
    }
}

int main() {
    printGpuInfo();
    return 0;
}
