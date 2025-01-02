#include<cuda_runtime.h>
#include<stdio.h>

__global__ void print_helloworld(){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = threadIdx.x+blockIdx.x*blockDim.x;
    printf("Hello world from block %d and thread %d id %d on gpu!\n",bid,tid,id);
}

int main(void){
    print_helloworld<<<2,3>>>();
    cudaDeviceSynchronize();
    return 0;
}