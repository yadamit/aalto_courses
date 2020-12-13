#include "cp.h"
#include<cmath>
#include <new>
#include<cstdlib>
#include <x86intrin.h>
#include<iostream>


typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));
static float8_t* float8_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(float8_t), sizeof(float8_t) * n)) {
        throw std::bad_alloc();
    }
    return (float8_t*)tmp;
}

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

float hsum8(float8_t x){
    float sum = 0;
    for(int i=0; i<8; i++){
        sum += x[i];
    }
    return sum;
}

void correlate(int ny, int nx, const float* data, float* result) {
    // TODO
    const int nb = 8;
    int na = (ny-1)/nb + 1;
    // int nab_v = na_v*nb_v;
    float8_t* d = float8_alloc(na*nx);

    double mu[ny] = {0};
    double std[ny] = {0};

    // calculating mean and std for each row
    #pragma omp parallel for schedule(static, 1)
    for(int row=0; row<ny; row++){
        for(int j=0; j<nx; j++){
            float tmp;
            tmp = (double)data[row*nx+j];
            mu[row] += tmp;
            std[row] += tmp*tmp;
        }
        mu[row] /= nx;
        std[row] = sqrt(std[row] - nx*mu[row]*mu[row]);
        // Note: std above = sqrt(nx)*actual_std
        // This saves us from dividing by nx after taking dot product.
    }

    // copy the data into new vectorized array
    #pragma omp parallel for schedule(static, 1)
    for(int ja=0; ja<na; ja++){
        for(int i=0; i<nx; i++){
            for(int kb=0; kb<nb; kb++){
                int row = ja*nb+kb;
                d[ja*nx + i][kb] = row<ny? ((double)data[row*nx + i] - mu[row])/std[row] : 0;
                // d[row*na_v + ka][kb] = (ka*nb_v +kb)<nx? ((float)data[row*nx + ka*nb_v +kb]-mu[row])/std[row] : 0;
            }
        }
    }

    #pragma omp parallel for schedule(static, 1)
    for(int ia=0; ia<na; ia++){
        for(int ja=ia; ja<na; ja++){
            float8_t dot000={0}, dot001={0}, dot010={0}, dot011={0}, dot100={0}, dot101={0}, dot110={0}, dot111={0};
            for(int k=0; k<nx; k++){
                float8_t a000 = d[ia*nx+k];
                float8_t b000 = d[ja*nx+k];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);
                dot000 += a000 * b000;
                dot001 += a000 * b001;
                dot010 += a010 * b000;
                dot011 += a010 * b001;
                dot100 += a100 * b000;
                dot101 += a100 * b001;
                dot110 += a110 * b000;
                dot111 += a110 * b001;
            }
            float8_t dot[8] = {dot000, dot001, dot010, dot011, dot100, dot101, dot110, dot111};
            for (int kb = 1; kb < 8; kb += 2) {
                dot[kb] = swap1(dot[kb]);
            }
            // for(int tmp=0; tmp<4; tmp++){
            //     printf("dot[%d]: %lf, %lf, %lf, %lf\n", tmp, dot[tmp][0], dot[tmp][1], dot[tmp][2], dot[tmp][3]);
            // }

            for(int ib=0; ib<nb; ib++){
                for(int jb=0; jb<nb; jb++){
                    int i = ia*nb + ib;
                    int j = ja*nb + jb;
                    if(i<=j && i<ny && j<ny){
                        result[i*ny+j] = (float)dot[jb^ib][jb];
                    }
                }
            }
            
            
        }
    }
    std::free(d);
}