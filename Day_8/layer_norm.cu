#include<iostream>
#include<stdio.h>
#include<cmath>

#define CHECK(call){        \
    const cudaError_t error = call; \
    if(error != cudaSuccess){   \
        printf("Error: %s, %d\n", __FILE__, __LINE__);  \
        printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error);    \
    }   \
}   \

__global__ void layerNorm(const float *A, float *B, int nx, int ny){
    //step 1: Calculate the row index according to the global memory.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    // step 2: do a condition so that index does not exceed the index of matrix.
    if (ix<nx){
        //step 3: setup shared memory for row-wise(nx) computations indicating un-size array using extern.
        extern __shared__ float shared[];
        float *nx_data = shared;

        //step 4: copy row-wise(nx) data to shared memory.
        for (int iy= threadIdx.y; iy<ny; iy+= blockDim.y){
            nx_data[iy] = A[ix*ny+iy];
        }
        __syncthreads();

        //step 5: compute the mean of the copied row.
        float mean = 0.0f;
        for(int y = 0; y<ny; ++y){
            mean += nx_data[y];
        }
        mean /= ny;

        //step 6: Compute variance.
        float variance = 0.0f;
        for (int y=0; y<ny; ++y){
            variance += (nx_data[y]-mean)*(nx_data[y]-mean); 
        }
        variance /= ny;
        float stddev = sqrtf(variance + 1e-7);

        //step 7: Normalize
        for(int y = threadIdx.y; y<ny; y+=blockDim.y){
            B[ix*ny+y] = (nx_data[y] - mean)/stddev;
        }

    }


}

int main(int argc, char **argv){
    printf("%s starting...\n",argv[0]);

    //step 8: setup the device.
    int dev= 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Using device %d:%s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //step 9: initialize the matrix setup params.
    const int nx=1<<8, ny=1<<8;
    float *h_A, *h_B;
    int nxy = nx*ny;
    int nByte = nxy*sizeof(float);

    //step 10: allocate memory on the host.
    h_A = (float*)malloc(nByte);
    h_B = (float*)malloc(nByte);

    //step 11: populate the matrix.
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            h_A[i * ny + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    //step 12: Allocate the device memory.
    float *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, nByte));
    CHECK(cudaMalloc((void **)&d_B,nByte));

    //step 13: copy data from host to device.
    CHECK(cudaMemcpy(d_A,h_A, nByte, cudaMemcpyHostToDevice));

    //step 14: setup the kernel parms.
    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x);
    size_t sharedMemSize = ny*sizeof(float);

    //step 15: launch the kernel.
    layerNorm<<<grid, block, sharedMemSize>>>(d_A,d_B,nx,ny);
    cudaDeviceSynchronize();

    //step 16: copy the results back the host.
    CHECK(cudaMemcpy(h_B,d_B, nByte, cudaMemcpyDeviceToHost));

    //step 17: Print results
    printf("A:\n");
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

    //step 18: Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;

}