#include <iostream>
#include <stdio.h>

#define CHECK(call){        \
    const cudaError_t error = call;  \
    if (error != cudaSuccess){          \
        printf("Error: %s, %d\n", __FILE__, __LINE__);  \
        printf("code %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(-10*error);    \
    }                       \
}                       \

__global__ void matrixVecMult(const float *A, const float *B, float *C, int N){

    // step 1: Set N as the number of columns of the matrix A = rows of vector B.

    // step 2: set the global thread index in the row major order.

        unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
        if (ix<N){
            float sum = 0.0f;
            for (int j=0; j<N; ++j){
                sum += A[ix*N+j] * B[j];
            }
            C[ix] = sum;
        }
}

int main(int argc, char **argv){
    printf("%s starting...\n",argv[0]);
    //step 3: setup the device.
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Compute Capibility %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Using device: %d:%s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //step 4: Initialize the matrix.
    const int nx = 256;
    int nxy = nx*nx;
    int nnByte = nxy* sizeof(float);
    int nByte = nx*sizeof(float);
    float *h_A, *h_B, *h_C;

    // step 5: allocate memory to host matrices.
    h_A = (float *)malloc(nnByte);
    h_B = (float *)malloc(nByte);
    h_C = (float *)malloc(nByte);

    // step 6: populate the matices:
    for (int i=0;i<nx;++i){
        for (int j=0;j<nx; ++j){
            h_A[i*nx+j] = 1.0f;
        }
        h_B[i] = 2.0f;
        h_C[i] = 0.0f;
    }

    // step 7: Allocate device memory.
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nnByte));
    CHECK(cudaMalloc((void **)&d_B, nByte));
    CHECK(cudaMalloc((void **)&d_C, nByte));

    // step 8: copy host data to device.
    CHECK(cudaMemcpy(d_A, h_A, nnByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nByte, cudaMemcpyHostToDevice));

    //step 9: configure the kernel.
    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x);

    // step 10: launch the kernel;
    matrixVecMult<<<grid,block>>>(d_A, d_B, d_C, nx);

    // step 11: copy the results back to host.
    CHECK(cudaMemcpy(h_C, d_C, nByte,cudaMemcpyDeviceToHost));

    // step 12: print the input and results.
    printf("A\n");
    for (int i=0; i<nx;++i){
        for (int j=0; j<nx; ++j){
            printf("%.2f", h_A[i*nx+j]);
        }
        printf("\n");
    }

    printf("B\n");
    for (int i =0; i<nx;++i){
        printf("%.2f", h_B[i]);
    }

    printf("\n");

    printf("C\n");
    for (int i = 0; i<nx; ++i){
        printf("%.2f", h_C[i]);
    }

    printf("\n");

    //step 13: cleanup.

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    CHECK(cudaDeviceReset());

    return 0;
}





