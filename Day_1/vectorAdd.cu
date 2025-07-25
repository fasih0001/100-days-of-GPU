#include <iostream>

__global__ void vecAdd(const float *a, const float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        c[i] = a[i] + b[i];
    }

}

int main(){
    const int N = 10;
    float A[N], B[N], C[N];

    float *d_a, *d_b,*d_c;
    cudaMalloc(&d_a,N*sizeof(float));
    cudaMalloc(&d_b,N*sizeof(float));
    cudaMalloc(&d_c,N*sizeof(float));
    cudaMemcpy(d_a,A,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,N*sizeof(float),cudaMemcpyHostToDevice);
    int blocksize=256;
    int gridsize=ceil(N/blocksize);
    vecAdd<<<gridsize,blocksize>>>(d_a,d_b,d_c,N);
    cudaMemcpy(C,d_c,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}