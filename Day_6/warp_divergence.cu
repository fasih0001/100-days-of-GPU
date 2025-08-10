#include <stdio.h>
                                    
#define CHECK(call){                \
    const cudaError_t error = call; \
    if (error != cudaSuccess){      \
        printf("Error : %s : %d\n", __FILE__, __LINE__);    \
        printf("code %d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(-10*error);    \
    }   \
}

// step one: write a naive RELU activation kernel.

__global__ void reluDivergent(float *in, float *out, int nx){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <nx){
        // In this case the a group of threads migh take of if branch and others might take the else brance causing divergence.
        // Note: warp divergence only happens within a single warp, Threads in different warps do not cause divergence.
        if (in[idx]> 0.0f){
            out[idx] = in[idx];
        }
        else{
            out[idx] = 0.0f;
        }
    }
}

__global__ void reluNonDivergent(float *in, float *out, int nx){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<nx){
        // Avoid branches but using the built in max function(compiles to a single instruction)
        out[idx]= fmaxf(in[idx],0.0f);
    }
}


int main(int argc, char **argv){

    printf("%s starting ... \n", argv[0]);

    // step 1: setup the device.
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Compute Capability %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Using Device %d : %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 20; // 2^20 means 1,048,576 values
    size_t nByte = nx*sizeof(float);

    // step 2: allocate the variable to memeory on host.
    float *h_in = (float*)malloc(nByte);
    float *h_out = (float*)malloc(nByte);

    // step 3: initialize mixed positive and negative values.
    for (int i=0; i<nx; ++i){
        h_in[i] = (i % 2 == 0) ? (float)i*0.001f : -(float)i*0.001f;
    }

    //step 3: Allocate the memory on the device.
    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, nByte));
    CHECK(cudaMalloc((void**)&d_out, nByte));

    // step 4: copy the memory from host to device.
    CHECK(cudaMemcpy(d_in, h_in, nByte, cudaMemcpyHostToDevice));

    // Step 5: setup the block and grid configuration.
    dim3 block(256); //A block of 256 threads.
    dim3 grid((nx+block.x-1)/block.x);

    // Step 5: run the divergent kernel.
    //reluDivergent<<<grid,block>>>(d_in, d_out, nx);
    reluNonDivergent<<<grid,block>>>(d_in, d_out, nx);
    CHECK(cudaDeviceSynchronize());

    // step 6: Copy the result from device to host.
    CHECK(cudaMemcpy(h_out, d_out, nx, cudaMemcpyDeviceToHost));

    // step 7: view the results.
    for (int i=0; i<10;++i){
        printf("h_out[%d]: %f\n", i, h_out[i]);
    }

    // step 7: cleanup.
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    CHECK(cudaDeviceReset());

    return 0;

}


