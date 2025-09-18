#include<stdio.h>
#include<iostream>
#include<vector>
#include<random>

//step1: cuda error checking.
#define CHECK(call){    \
    const cudaError_t error = call; \
    if(error != cudaSuccess){   \
        printf("Error: %s:%d\n",__FILE__, __LINE__);    \
        printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
    }   \
}   \

__global__ void softMax(const float *__restrict__ A, float *__restrict__ B, int nx, int ny){
    //step3 : global indexing for rows, so we compute softmax for each row of the matrix.
    //resulting matrix B is the (ny,1)
    //To get rid of confusion:
    // Row index   iy=0   iy=1   iy=2   iy=3 -> column index
    //  |
    //  V
    // ix=0        [                    ]  
    // ix=1        [                    ]  
    // ix=2        [                    ]  

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix<nx){
        float xMax = -INFINITY;
        float norm = 0.00f;

        //step 4: Pass 1 - fused max + Norm.
        for (int iy = 0; iy<ny; ++iy){
            float curr = A[ix*ny+iy];       
            if (curr> xMax){
                norm = norm * expf(xMax-curr);
                xMax = curr;
            }
            norm += expf(curr -xMax);
        }

        //step 5: Pass 2: Normalize.
        for (int iy=0; iy<ny; ++iy){
            B[ix*ny+iy]= expf(A[ix*ny+iy] - xMax)/ norm;
        }
    }

}

int main(int argc, char **argv){
    printf("%s Starting ...\n", argv[0]);

    // step 6: setup the device.
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Device: %d: %s", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //step 7: initialize the matrix setup params.
    const int nx=1<<3, ny=1<<3;
    float *h_A, *h_B;
    int nxy = nx*ny;
    int nByte = nxy*sizeof(float);

    //step 8: allocate memory on the host.
    h_A = (float*)malloc(nByte);
    h_B = (float*)malloc(nByte);

    //step 9: populate the matrix.
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            h_A[i * ny + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    //step 10: Allocate the device memory.
    float *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, nByte));
    CHECK(cudaMalloc((void **)&d_B,nByte));

    //step 11: copy data from host to device.
    CHECK(cudaMemcpy(d_A,h_A, nByte, cudaMemcpyHostToDevice));

    //step 12: setup the kernel parms.
    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x);

    //step 13: launch the kernel.
    softMax<<<grid, block>>>(d_A,d_B,nx,ny);
    cudaDeviceSynchronize();

    //step 14: copy the results back the host.
    CHECK(cudaMemcpy(h_B,d_B, nByte, cudaMemcpyDeviceToHost));

    //step 15: Print results
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

    //step 16: Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}