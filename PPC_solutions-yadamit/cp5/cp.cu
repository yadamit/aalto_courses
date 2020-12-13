#include "cp.h"
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "stopwatch.h"
// #include<vector>

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

__global__ void mykernel(float* t, float* r, int ny, int nx, int naby, int nabx) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bx_64 = bx*64;
    int by_64 = by*64;
    // if(tx==0 and ty==0){
    //     printf("First element: %f, %f   \n", t[0], d[0]);
    //     printf("Blockidx: %d %d\n", bx, by);
    // }

    float dot[8][8];
    // initialize dot
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            dot[i][j]=  0;
        }
    }
    for(int k=0; k<nx; k++){
        float x[8], y[8]; //x refers to row values
        // read x and y
        for(int i=0; i<8; i++){
            int idx = bx_64 + i*8 + tx;
            x[i] = t[k*naby + idx];
        }
        for(int i=0; i<8; i++){
            int idx = by_64 + i*8 + ty;
            y[i] = t[k*naby + idx];
        }
        for(int i=0; i<8; i++){
            for(int j=0; j<8; j++){
                dot[i][j] += x[j]*y[i];
            }
        }
    }
    // if(tx==0 and ty==0){
    //     printf("dot[0]:%f\n",dot[0]);
    // }

    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            int id_x = bx_64 + j*8 + tx;
            int id_y = by_64 + i*8 + ty;
            if(id_x<ny && id_y<ny)
                r[id_y*ny + id_x] = dot[i][j];
        }
    }
    // if(tx==0 and ty==0){
    //     printf("r[0]: %f, dot[0]: %f\n", r[0], dot[0]);
    // }

    // for(int col=0; col<nx; col++){
    //     dot += d[i*nx+col]*d[j*nx+col];
    // }
    // r[i*ny+j] = dot;
}

__global__ void myppkernel(float* d, float* t, int ny, int nx, int naby, int nabx, double* mu, double* std){
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    

    double m = mu[bx], s = std[bx];
    for(int i=tx; i<nabx; i+=64){
        t[i*naby+bx] = (bx<ny and i<nx)? (float)(d[bx*nx + i]-m)/s : 0;
    }
    
    // if(tx==0 and bx==0){
    //     printf("mu[0]:%lf, std[0]:%lf\n", mu[0], std[0]);
    //     printf("t[0]:%f\n", t[0]);
    //     printf("d[0]:%f\n", d[0]);
    // }

}

void correlate(int ny, int nx, const float* data, float* result) {
    // FIXME
    ppc::stopwatch sw;

    const int nb = 64;
    int nax = (nx-1)/nb + 1;
    int nay = (ny-1)/nb + 1;
    int nabx = nax*nb;
    int naby = nay*nb;
    // std::vector <float> d(nax*nb*nay*nb, 0);
    // std::vector <float> t(nax*nb*nay*nb, 0);
    // float* t  = NULL;
    // t = (float*)calloc(nabx*naby,sizeof(float));

    double mu[ny] = {0};
    double std[ny] = {0};
    
    // #pragma omp parallel for schedule(static, 1)
    sw.record();
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
    sw.record();

    // copy data in a new padded array
    // for(int i=0; i<nx; i++){
    //     for(int j=0; j<ny; j++){
    //         t[i*naby+j] = (float)(data[j*nx+i]-mu[j])/std[j];
    //     }
    // }
    sw.record();
    // printf("d:\n");
    // for(int i=0; i<naby/3; i++){
    //     for(int j=0; j<nabx/3; j++){
    //         printf("%f ", d[i*nabx+j]);
    //     }
    //     printf("\n");
    // }
    // printf("t:\n");
    // for(int i=0; i<naby/3; i++){
    //     for(int j=0; j<nabx/3; j++){
    //         printf("%f ", t[i*naby+j]);
    //     }
    //     printf("\n");
    // }
    // normalizing the input
    // float* d = NULL;
    // d = (float*)malloc(ny*nx*sizeof(float));
    // // #pragma omp parallel for schedule(static, 1)
    // for(int row=0; row<ny; row++){
    //     for(int j=0; j<nx; j++){
    //         d[nx*row+j] = (float)(data[nx*row+j]-mu[row])/std[row];
    //     }
    // }

    float* tGPU = NULL;
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny*nx*sizeof(float)));
    CHECK(cudaMemcpy(dGPU, data, ny*nx*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&tGPU, naby * nabx * sizeof(float)));

    double * m = NULL;
    double * s = NULL;
    CHECK(cudaMalloc((void**)&m, ny*sizeof(double)));
    CHECK(cudaMalloc((void**)&s, ny*sizeof(double)));
    CHECK(cudaMemcpy(m, mu, ny*sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(s, std, ny*sizeof(double), cudaMemcpyHostToDevice));
    myppkernel<<<naby, 64>>>(dGPU, tGPU, ny, nx, naby, nabx, m, s); //ny blocks, 64 threads in each block
    CHECK(cudaGetLastError());
    // CHECK(cudaMemcpy(t, tGPU, naby * naby * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));

    float* rGPU = NULL; 
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    // CHECK(cudaMemcpy(tGPU, t, naby * nabx * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(8, 8); //16 is the #threads in one block
    dim3 dimGrid(divup(naby, dimBlock.x*8), divup(naby, dimBlock.y*8));
    mykernel<<<dimGrid, dimBlock>>>(tGPU, rGPU, ny, nx, naby, nabx);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(tGPU));
    CHECK(cudaFree(rGPU));

    sw.record();
    // sw.print();
}
