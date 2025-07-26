#include <iostream>
#include <stdio.h>

__global__ void checkindex(void){

    printf("threadidx: (%d, %d, %d) | blockidx: (%d, %d, %d) | blockdim: (%d, %d, %d) | griddim: (%d, %d, %d)\n", threadIdx.x, threadIdx.y,
    threadIdx.z, blockIdx.x, blockIdx.y,blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z );

}

int main( int argc, char **argv){
    //step 1: define the data size to be used in the program
    int nElem = 6;

    //step 2: define the block size and calculate the grid size based on the block and data size
    dim3 block(3);
    dim3 grid((nElem + block.x -1)/block.x);

    // step 3 : check grid and block dimension from host side
    printf("block.x= %d, block.y= %d, block.z= %d \n", block.x, block.y, block.z);
    printf("grid.x= %d, grid.y = %d, grid.z= %d \n", grid.x, grid.y, grid.z);

    // check grid and block dimension from device side
    checkindex <<< grid, block >>> ();

    // reset the device before you leave
    cudaDeviceReset();

    return(0);

}

