#include <iostream>
#include <stdio.h>

// step 1: a validation h
#include <cstdlib>  // for rand()
#include <iomanip>

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){

    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy=0; iy<ny; iy++){

        for (int ix=0; ix<nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;

    }

}


int main(int argc, char **argv){

    int nx = 4;
    int ny = 3;
    int size = nx * ny;

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size];

    std::cout << "Matrix A:\n";
    for (int i = 0; i < size; ++i) {
        A[i] = static_cast<float>(std::rand() % 10);
        std::cout << std::setw(5) << A[i];
        if ((i + 1) % nx == 0) std::cout << "\n";
    }

    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < size; ++i) {
        B[i] = static_cast<float>(std::rand() % 10);
        std::cout << std::setw(5) << B[i];
        if ((i + 1) % nx == 0) std::cout << "\n";
    }

    // Call the function
    sumMatrixOnHost(A, B, C, nx, ny);

    // Print the result
    std::cout << "Result matrix C:\n";
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            std::cout << std::setw(5) << C[iy * nx + ix] << " ";
        }
        std::cout << "\n";
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;

}