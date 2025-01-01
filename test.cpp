#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>  // 用于sleep函数
#include <my_infer.h>



// CUDA 核函数，执行向量加法
__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    int N = 1000;  // 向量大小
    int size = N * sizeof(int);

    // 在主机（CPU）上分配内存
    int* h_A = new int[N];
    int* h_B = new int[N];
    int* h_C = new int[N];

    // 初始化向量
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // 在设备（GPU）上分配内存
    int* d_A;
    int* d_B;
    int* d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 计算 block 和 grid 大小
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 启动 CUDA 核函数
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 检查核函数是否执行成功
    cudaDeviceSynchronize();

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    std::cout << "Waiting for 5 seconds to monitor GPU..." << std::endl;
    sleep(5);  // 等待 5 秒

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
