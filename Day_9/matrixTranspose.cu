#include<iostream>
#include<stdio.h>

// step1: Cuda error handling.
#define CHECK(call){    \
    const cudaError_t error = call; \
    if(error != cudaSuccess){  \
        printf("Error: %s, %d\n", __FILE__, __LINE__);  \
        printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error));   \
        return(-10*error);  \
    }   \
}   \

__global__ void matrixTranspose(const float *A, float *B, int nx, int ny){

    //step 2: calculate the cuda global index values.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix< nx && iy<ny){
        //step 3: Read the columns and write the rows. Effectively uses L1 Cache therefore improved cache hits as Read ops can be cached.
        B[iy*nx+ix] = A[ix*ny+iy];
    }
}

int main(int argc, char **argv){
    printf("%s starting...\n",argv[0]);

    //step 4: Setup the device.
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Device: %d:%s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //step 5: initialize the matrix setup params.
    const int nx=1<<3, ny=1<<3;
    float *h_A, *h_B;
    int nxy = nx*ny;
    int nByte = nxy*sizeof(float);

    //step 6: allocate memory on the host.
    h_A = (float*)malloc(nByte);
    h_B = (float*)malloc(nByte);

    //step 7: populate the matrix.
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            h_A[i * ny + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    //step 8: Allocate the device memory.
    float *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, nByte));
    CHECK(cudaMalloc((void **)&d_B,nByte));

    //step 9: copy data from host to device.
    CHECK(cudaMemcpy(d_A,h_A, nByte, cudaMemcpyHostToDevice));

    //step 10: setup the kernel parms.
    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x);

    //step 11: launch the kernel.
    matrixTranspose<<<grid, block>>>(d_A,d_B,nx,ny);
    cudaDeviceSynchronize();

    //step 12: copy the results back the host.
    CHECK(cudaMemcpy(h_B,d_B, nByte, cudaMemcpyDeviceToHost));

    //step 13: Print results
    printf("\nA:\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%.2f ", h_A[i * ny + j]);
        }
        printf("\n");
    }

    printf("\nB:\n");
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            printf("%.2f ", h_B[i * ny + j]);
        }
        printf("\n");
    }

    //step 14: Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}