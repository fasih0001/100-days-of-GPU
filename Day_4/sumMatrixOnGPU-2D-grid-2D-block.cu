#include <iostream>
#include <stdio.h>


#define CHECK(call)                                                      \
{                                                                        \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
        printf("Error: %s : %d\n", __FILE__, __LINE__);                  \
        printf("code %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10 * error);                                               \
    }                                                                    \
}

// Kernel to perform element-wise matrix addition
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy + ix * ny;

    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

// Function to initialize matrix data
void initialiseData(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand() % 10);  // random integers 0-9
    }
}

// Function to print a matrix block
void printMatrixBlock(const char* label, float *matrix, int nx, int ny, int maxRows = 5, int maxCols = 5) {
    printf("\n%s (Top-left %d x %d block):\n", label, maxRows, maxCols);
    for (int i = 0; i < maxRows; ++i) {
        for (int j = 0; j < maxCols; ++j) {
            printf("%4.1f ", matrix[i * ny + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // Step 1: Setup the Device.
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Step 2: Setup matrix dimensions.
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nByte = nxy * sizeof(float);
    printf("Matrix size %d by %d\n", nx, ny);

    // Step 3: Allocate memory on Host.
    float *h_A, *h_B, *gpuResult;
    h_A = (float *)malloc(nByte);
    h_B = (float *)malloc(nByte);
    gpuResult = (float *)malloc(nByte);

    // Step 3.1: Initialize host matrices
    srand(time(NULL));
    initialiseData(h_A, nxy);
    initialiseData(h_B, nxy);

    // Debug: Print top-left portion of A and B
    printMatrixBlock("Matrix A", h_A, nx, ny);
    printMatrixBlock("Matrix B", h_B, nx, ny);

    // Step 4: Allocate memory on Device.
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nByte));
    CHECK(cudaMalloc((void **)&d_B, nByte));
    CHECK(cudaMalloc((void **)&d_C, nByte));

    // Step 5: Copy host data to device.
    CHECK(cudaMemcpy(d_A, h_A, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nByte, cudaMemcpyHostToDevice));

    // Step 6: Launch the kernel.
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());

    // Step 7: Copy result back to host.
    CHECK(cudaMemcpy(gpuResult, d_C, nByte, cudaMemcpyDeviceToHost));

    // Debug: Print top-left portion of result matrix C
    printMatrixBlock("Matrix C = A + B", gpuResult, nx, ny);

    // Step 8: Cleanup
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(gpuResult);
    CHECK(cudaDeviceReset());

    return 0;
}
