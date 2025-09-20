#include<iostream>
#include<stdio.h>
#include<vector>
#include<random>
#include <thrust/device_vector.h>

//step 0: setup cuda error checking.
#define CHECK(call){    \
    const cudaError_t error = call; \
    if(error != cudaSuccess){   \
        printf("Error: %s:%d\n", __FILE__, __LINE__);   \
        printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        return(-10*error);   \
    }   \
}   \


//step1: CEIL_DIV to create as many blocks as necessary to map all of C.
//Constexpr tells the compiler that the function can be evaluated at compile time
//if all the inputs are known at compile time.
constexpr int CEIL_DIV(int a, int b) {
    return (a + b - 1) / b;
}

//step 2: cuda kernel. The matrix is [nx x nk]* [nk x ny] = [nx x ny].
//Here each thread does a single multiplication between two elements.
//Each thread works independently therefore no thread synchrinization is required(Asynchronous Kernel execution).
__global__ void sgemm_naive(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,

    int nx,
    int ny,
    int nk,

    float alpha,
    float beta
){
    //setp 3: setup the global indexing.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    //step 4: The basic of row major indexing to multiply a row of A with col of B.
    //The row major elements e.g A[row][col] are stored as " row*num_col+col ".
    if(ix<nx && iy<ny){
        float tmp = 0.0;
        for (int i=0; i<nk; ++i){
            tmp += A[ix*nk+i]*B[i*ny+iy];
        }

        //step 5: As per BLAS implementation of SGEMM : C = alpha*(A@B)+ beta*C
        C[ix*ny+iy] = alpha * tmp + beta * C[ix*ny+iy];
    }
}

void printMatrix(const std::vector<float>& mat, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char **argv){
    std::cout<< argv[0]<< "Starting...\n";

    //step 6: setup the device.
    int dev=0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Device %d:%s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //step 7: setup the matrix dims and params.
    int nx = 1<<8;
    int nk = 1<<8;
    int ny = 1<<8;

    float alpha = 1.0f;
    float beta = 0.0f;

    //step 8: setup the host matrices.
    std::vector<float> h_A(nx*nk);
    std::vector<float> h_B(nk*ny);
    std::vector<float> h_C(nx*ny);

    //step 9: random number generation.
    std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_real_distribution<float> dist(0.0f,1.0f);

    //This modern for loop is like in python: " for element in list " where list is a container.
    //where the data type of element is self determined in python (C++ uses auto keyword instead).
    for (auto& val : h_A) val = dist(rng);
    for (auto &val : h_B) val = dist(rng);

    //step 10: setup device matrices : RAII using Thrust::device_vector.
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B =h_B;
    thrust::device_vector<float> d_C(nx*ny,0.0f); //elements of the resulting C vector are set to 0.

    //step 11: kernel launch params.
    dim3 blockThreads(32,32,1); //BlockDim
    dim3 numBlocks(CEIL_DIV(nx,32),CEIL_DIV(ny,32),1); //gridDim

    //step 12: launch kernel
    sgemm_naive<<<numBlocks,blockThreads>>>(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()),
        nx,ny,nk,alpha,beta
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    //step 13: copy the result back to host.
    //syntax: thrust::copy(source_begin, source_end, dest_begin);
    thrust::copy(d_C.begin(),d_C.end(),h_C.begin());

    //step 14: Print a few elements for verification
    // printMatrix(h_A, nx, nk, "A");
    // printMatrix(h_B, nk, ny, "B");
    // printMatrix(h_C, nx, ny, "C");

    std::cout << "Done!\n";
    return 0;
}


