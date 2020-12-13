#include <cstdio>
#include <cmath>
#include "nn.h"
#include <iostream>
#include <cuda_runtime.h>
#include "stopwatch.h"
// ------------------------------------------------------------------------

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
#define CHECK(x) check(x, #x);

float* g_weights = NULL;    // store all network weights in one big array.

// ------------------------------------------------------------------------

ConvLayer g_convLayers[16] = {
    { 224,  64,   3,        0,     1728 },
    { 224,  64,  64,     1792,    38656 },    // 2x2 maxpool (224 x 224 -> 112 x 112)
    { 112, 128,  64,    38720,   112448 },
    { 112, 128, 128,   112576,   260032 },    // 2x2 maxpool (112 x 112 -> 56 x 56)
    {  56, 256, 128,   260160,   555072 },
    {  56, 256, 256,   555328,  1145152 },
    {  56, 256, 256,  1145408,  1735232 },
    {  56, 256, 256,  1735488,  2325312 },    // 2x2 maxpool (56 x 56 -> 28 x 28)
    {  28, 512, 256,  2325568,  3505216 },
    {  28, 512, 512,  3505728,  5865024 },
    {  28, 512, 512,  5865536,  8224832 },
    {  28, 512, 512,  8225344, 10584640 },    // 2x2 maxpool (28 x 28 -> 14 x 14)
    {  14, 512, 512, 10585152, 12944448 },
    {  14, 512, 512, 12944960, 15304256 },
    {  14, 512, 512, 15304768, 17664064 },
    {  14, 512, 512, 17664576, 20023872 },    // 2x2 maxpool (14 x 14 -> 7 x 7) -> interpret as flat array
};

DenseLayer g_denseLayers[3] = {
    { 4096, 25088,  20024384, 122784832, false },
    { 4096,  4096, 122788928, 139566144, false },
    { 1000,  4096, 139570240, 143666240, true  },
};

// ------------------------------------------------------------------------

__global__ void evalConv(int idx, const float* bufIn, float* bufOut, float*g_weightsGPU, ConvLayer* g_convLayersGPU){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    // (0, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    const ConvLayer& layer = g_convLayersGPU[idx];
    const float* W = g_weightsGPU + layer.ofsW;
    const float* B = g_weightsGPU + layer.ofsB;
    
    // if (idx==0 and bx==0 and tx==0 and ty==0) printf("%d, %d, %d\n" ,idx, tx, ty);
    // printf("conv %-2d (%3d, %3d, %3d) -> (%3d, %3d, %3d)\n", idx, layer.nIn, layer.sz, layer.sz, layer.nOut, layer.sz, layer.sz);
    // fflush(stdout);

    int sz = layer.sz;
    // for (int i = 0; i < layer.nOut; i++)
    // for (int y = 0; y < sz; y++)
    // for (int x = 0; x < sz; x++)
    int i = bx;
    int y = by;
    int x = tx;
    {
        float sum = B[i];
        for (int j = 0; j < layer.nIn; j++)
        for (int dy = 0; dy < 3; dy++)
        for (int dx = 0; dx < 3; dx++){
            int yy = y + dy - 1;
            int xx = x + dx - 1;
            if (yy >= 0 && yy < sz && xx >= 0 && xx < sz)
                sum += bufIn[sz*sz*j + sz*yy + xx] * W[layer.nIn*3*3*i + 3*3*j + 3*(2-dy) + (2-dx)];
        }
        bufOut[sz*sz*i + sz*y + x] = (sum > 0.f) ? sum : 0.f; // ReLu activation.
    }
}

// ------------------------------------------------------------------------

__global__ void evalDense(int idx, const float* bufIn, float* bufOut, float*g_weightsGPU, DenseLayer* g_denseLayersGPU, float* total)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    const DenseLayer& layer = g_denseLayersGPU[idx];
    const float* W = g_weightsGPU + layer.ofsW;
    const float* B = g_weightsGPU + layer.ofsB;
    // float total = 0.f;

    // printf("dense %d (%3d) -> (%3d)\n", idx, layer.nIn, layer.nOut);
    // fflush(stdout);

    int i = bx*blockDim.x + tx;
    
    // for (int i = 0; i < layer.nOut; i++)
    if(i>=layer.nOut) return;
    // {
        float sum = B[i];
        for (int j = 0; j < layer.nIn; j++)
            sum += bufIn[j] * W[layer.nIn*i + j];

        if (layer.softmax)
            // total += (bufOut[i] = expf(sum));
            atomicAdd(total, (bufOut[i] = expf(sum)));
        else
            bufOut[i] = (sum > 0.f) ? sum : 0.f;
    // }

    // if (layer.softmax)
    //     for (int i = 0; i < layer.nOut; i++)
    //         bufOut[i] *= 1.f / total;
}


__global__ void denseSoftmax(int idx, const float* bufIn, float* bufOut, float* total, DenseLayer* g_denseLayersGPU){
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    const DenseLayer& layer = g_denseLayersGPU[idx];

    int i = bx*blockDim.x + tx;
    if(i>=layer.nOut) return;

    bufOut[i] = bufIn[i] / (*total);
}

// ------------------------------------------------------------------------

#define MAX(a, b) ((a) > (b) ? (a) : (b))
__global__ void maxPool2x2(int sz, int n, const float* bufIn, float* bufOut)
{
    // printf("maxpool (%3d, %3d, %3d) -> (%3d, %3d, %3d)\n", n, sz, sz, n, sz/2, sz/2);
    // fflush(stdout);
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int h = sz >> 1;
    int i=bx, x=tx;
    // for (int i = 0; i < n; i++)
    for (int y = 0; y < h; y++)
    // for (int x = 0; x < h; x++)
    {
        float v0 = bufIn[sz*sz*i + sz*(y*2)   + (x*2)];
        float v1 = bufIn[sz*sz*i + sz*(y*2)   + (x*2+1)];
        float v2 = bufIn[sz*sz*i + sz*(y*2+1) + (x*2)];
        float v3 = bufIn[sz*sz*i + sz*(y*2+1) + (x*2+1)];
        bufOut[i*h*h + x + h*y] = MAX(MAX(MAX(v0, v1), v2), v3);
    }
}

// ------------------------------------------------------------------------

void evalNetwork(float *buf0) {
    float* buf1 = new float[64 * 224 * 224];

    // Evaluate the network, ping-pong data between buffers.
    printf("Starting inference.\n");
    fflush(stdout);

    float *g_weightsGPU = NULL;
    ConvLayer *g_convLayersGPU = NULL;
    DenseLayer *g_denseLayersGPU = NULL;
    CHECK(cudaMalloc((void**)&g_weightsGPU, 143667240*sizeof(float)));
    CHECK(cudaMemcpy(g_weightsGPU, g_weights, 143667240*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&g_convLayersGPU, 16*sizeof(ConvLayer)));
    CHECK(cudaMemcpy(g_convLayersGPU, g_convLayers, 16*sizeof(ConvLayer), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&g_denseLayersGPU, 3*sizeof(DenseLayer)));
    CHECK(cudaMemcpy(g_denseLayersGPU, g_denseLayers, 3*sizeof(DenseLayer), cudaMemcpyHostToDevice));

    // copy input(buf0) to GPU and allocate space for output(buf1)
    int buf_size = 64*224*224;
    float* dGPU = NULL;
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, buf_size*sizeof(float)));
    CHECK(cudaMalloc((void**)&rGPU, buf_size*sizeof(float)));
    CHECK(cudaMemcpy(dGPU, buf0, buf_size*sizeof(float), cudaMemcpyHostToDevice));

    // layer 0
    int layer = 0;
    dim3 threads(g_convLayers[layer].sz, g_convLayers[layer].sz);
    dim3 blocks(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 1
    layer = 1;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // DEBUG
    // CHECK(cudaMemcpy(buf1, dGPU, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
    // if(true){ //64 x 224 x 224
    //     // print first 10x10 of first output layer
    //     int sz = g_convLayers[layer].sz;
    //     int out_layer = 10;
    //     printf("Printing layer %d, sz=%d\n", out_layer, sz);
    //     for(int i=0; i<20; i++){
    //         for(int j=0; j<20; j++){
    //             printf("%f ", buf1[sz*sz*out_layer + sz*j + i]);
    //         }
    //         printf("\n");
    //     }
    //     // return;
    // }

    // maxPool2x2 layer 1
    // threads = dim3(224>>1, 224>>1);
    maxPool2x2<<<64, 112>>>(224, 64, dGPU, rGPU);

    // layer 2
    layer = 2;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 3
    layer = 3;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // maxPool2x2 layer 2
    // threads = dim3(112>>1, 112>>1);
    maxPool2x2<<<128, 56>>>(112, 128, rGPU, dGPU);

    // layer 4
    layer = 4;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 5
    layer = 5;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 6
    layer = 6;
    blocks =dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 7
    layer = 7;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // maxPool2x2 layer 3
    // threads = dim3(56>>1, 56>>1);
    maxPool2x2<<<256, 28>>>(56, 256, dGPU, rGPU);

    // layer 8
    layer = 8;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 9
    layer = 9;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();
    
    // layer 10
    layer = 10;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();
    
    // layer 11
    layer = 11;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // maxPool2x2 layer 4
    // threads = dim3(28>>1, 28>>1);
    maxPool2x2<<<512, 14>>>(28, 512, rGPU, dGPU);

    // layer 12
    layer = 12;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 13
    layer = 13;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 14
    layer = 14;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, dGPU, rGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // layer 15
    layer = 15;
    blocks = dim3(g_convLayers[layer].nOut,g_convLayers[layer].sz);
    evalConv<<<blocks, g_convLayers[layer].sz >>>(layer, rGPU, dGPU, g_weightsGPU, g_convLayersGPU);
    cudaDeviceSynchronize();

    // maxPool2x2 layer 5
    // threads = dim3(14>>1, 14>>1);
    maxPool2x2<<<512, 7>>>(14, 512, dGPU, rGPU);

    
    // total for softmax
    float* totalGPU = NULL;
    float zero[1] = {0};
    CHECK(cudaMalloc((void**)&totalGPU, 1 * sizeof(float)));
    CHECK(cudaMemcpy(totalGPU, zero, 1 * sizeof(float), cudaMemcpyHostToDevice));

    // dense 0
    CHECK(cudaMemcpy(totalGPU, zero, 1 * sizeof(float), cudaMemcpyHostToDevice)); //total=0
    evalDense<<<divup(g_denseLayers[0].nOut, 64), 64>>>(0, rGPU, dGPU, g_weightsGPU, g_denseLayersGPU, totalGPU);
    cudaDeviceSynchronize();
    if(g_denseLayers[0].softmax){
        denseSoftmax<<<divup(g_denseLayers[0].nOut, 32), 32>>>(0, dGPU, dGPU, totalGPU, g_denseLayersGPU);
    }
    
    // dense 1
    CHECK(cudaMemcpy(totalGPU, zero, 1 * sizeof(float), cudaMemcpyHostToDevice)); //total=0
    evalDense<<<divup(g_denseLayers[1].nOut, 64), 64>>>(1, dGPU, rGPU, g_weightsGPU, g_denseLayersGPU, totalGPU);
    cudaDeviceSynchronize();
    if(g_denseLayers[1].softmax){
        denseSoftmax<<<divup(g_denseLayers[1].nOut, 32), 32>>>(1, rGPU, rGPU, totalGPU, g_denseLayersGPU);
    }

    // dense 2
    CHECK(cudaMemcpy(totalGPU, zero, 1 * sizeof(float), cudaMemcpyHostToDevice)); //total=0
    evalDense<<<divup(g_denseLayers[2].nOut, 64), 64>>>(2, rGPU, dGPU, g_weightsGPU, g_denseLayersGPU, totalGPU);
    cudaDeviceSynchronize();
    if(g_denseLayers[2].softmax){
        denseSoftmax<<<divup(g_denseLayers[2].nOut, 64), 64>>>(2, dGPU, dGPU, totalGPU, g_denseLayersGPU);
    }
    cudaDeviceSynchronize();


    // DEBUG
    // {
    //     CHECK(cudaMemcpy(buf1, dGPU, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CHECK(cudaMemcpy(zero, totalGPU, 1* sizeof(float), cudaMemcpyDeviceToHost));
    //     // print first 100 of first output layer
    //     for(int i=0; i<20; i++){
    //         printf("%f ", buf1[i]);
    //     }
    //     printf("TOTAL: %f\n", zero[0]);
    // }





    CHECK(cudaMemcpy(buf0, dGPU, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    CHECK(cudaFree(g_convLayersGPU));
    CHECK(cudaFree(g_denseLayersGPU));
    CHECK(cudaFree(g_weightsGPU));
    CHECK(cudaFree(totalGPU));


    // evalDense<<<divup(g_denseLayers[0].nOut, 32), 32>>>(0, rGPU, dGPU, g_weightsGPU, g_denseLayersGPU);
    // cudaDeviceSynchronize();

    // threads = dim3(224>>1, 224>>1)
    // maxPool2x2<<<64, threads>>>(224, 64, dGPU, rGPU);


    // evalConv(0, buf0, buf1);
    // evalConv(1, buf1, buf0);
    // maxPool2x2(224, 64, buf0, buf1);
    // evalConv(2, buf1, buf0);
    // evalConv(3, buf0, buf1);
    // maxPool2x2(112, 128, buf1, buf0);
    // evalConv(4, buf0, buf1);
    // evalConv(5, buf1, buf0);
    // evalConv(6, buf0, buf1);
    // evalConv(7, buf1, buf0);
    // maxPool2x2(56, 256, buf0, buf1);
    // evalConv(8, buf1, buf0);
    // evalConv(9, buf0, buf1);
    // evalConv(10, buf1, buf0);
    // evalConv(11, buf0, buf1);
    // maxPool2x2(28, 512, buf1, buf0);
    // evalConv(12, buf0, buf1);
    // evalConv(13, buf1, buf0);
    // evalConv(14, buf0, buf1);
    // evalConv(15, buf1, buf0);
    // maxPool2x2(14, 512, buf0, buf1);
    // evalDense(0, buf1, buf0);
    // evalDense(1, buf0, buf1);
    // evalDense(2, buf1, buf0);

    printf("Done.\n\n");
    fflush(stdout);
    delete[] buf1;
}

// ------------------------------------------------------------------------