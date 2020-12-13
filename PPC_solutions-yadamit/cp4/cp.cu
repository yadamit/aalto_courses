#include "cp.h"
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

__global__ void mykernel(int ny, int nx, float* d, float* r) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i>=ny or j>=ny){
        return;
    }
    float dot = 0;
    for(int col=0; col<nx; col++){
        dot += d[i*nx+col]*d[j*nx+col];
    }
    r[i*ny+j] = dot;
}

void correlate(int ny, int nx, const float* data, float* result) {
    // FIXME
    double mu[ny] = {0};
    double std[ny] = {0};
    // #pragma omp parallel for schedule(static, 1)
    for(int row=0; row<ny; row++){
        for(int j=0; j<nx; j++){
            double tmp;
            tmp = (double)data[row*nx+j];
            mu[row] += tmp;
            std[row] += tmp*tmp;
        }
        mu[row] /= nx;
        std[row] = sqrt(std[row] - nx*mu[row]*mu[row]);
        // Note: std above = sqrt(nx)*actual_std
        // This saves us from dividing by nx after taking dot product.
    }

    // normalizing the input
    float* d = NULL;
    d = (float*)malloc(ny*nx*sizeof(float));
    // #pragma omp parallel for schedule(static, 1)
    for(int row=0; row<ny; row++){
        for(int j=0; j<nx; j++){
            d[nx*row+j] = (float)(data[nx*row+j]-mu[row])/std[row];
        }
    }

    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL; 
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16); //16 is the #threads in one block
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, rGPU);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));

    
}
