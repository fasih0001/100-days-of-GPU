// This code serves as the Testing-123 example to ensure that cuda code is correctly working on the GPU.

#include <iostream>

__global__ void hello_from_gpu() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    hello_from_gpu<<<1, 5>>>(); // One block of 5 threads.
    cudaDeviceSynchronize(); // synchronize to ensure the gpu finishes before exiting.
    return 0;
}