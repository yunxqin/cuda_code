#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);  // 获取可用的 GPU 数量
    if (deviceCount == 0) {
        std::cerr << "No CUDA capable devices found!" << std::endl;
        return -1;
    }

    // 选择第一个设备（你也可以根据需要选择其他设备）
    int deviceId = 0;
    cudaSetDevice(deviceId);

    // 获取 GPU 设备属性
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    // 输出一些设备信息
    std::cout << "GPU Name: " << deviceProp.name << std::endl;
    std::cout << "Total SMs: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;

    // 计算 GPU 上可以并行的最大线程数
    int totalThreads = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
    std::cout << "Total threads supported by the GPU: " << totalThreads << std::endl;

    return 0;
}
