#include <iostream>
#include <cuda_runtime.h>

void getMemoryInfo() {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    std::cout << "Free memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;
}

int main() {
    getMemoryInfo();
    return 0;
}
