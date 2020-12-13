#include "cp.h"
#include<cmath>
#include <new>
#include<cstdlib>
#include <x86intrin.h>
#include<iostream>


typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 0b00000001); }
// static inline double4_t swap2(double4_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b00000101); }

double hsum4(double4_t x){
    double sum = 0;
    for(int i=0; i<4; i++){
        sum += x[i];
    }
    return sum;
}

void correlate(int ny, int nx, const float* data, float* result) {
    // TODO
    // printf("Hello");
    const int nb = 4;
    int na = (ny-1)/nb + 1;
    na = (na%2==0)? na:na+1;
    // int nab_v = na_v*nb_v;
    double4_t* d = double4_alloc(na*nx);

    double mu[ny] = {0};
    double std[ny] = {0};

    // calculating mean and std for each row
    #pragma omp parallel for schedule(static, 1)
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

    // copy the data into new vectorized array
    #pragma omp parallel for schedule(static, 1)
    for(int ja=0; ja<na; ja++){
        for(int i=0; i<nx; i++){
            for(int kb=0; kb<nb; kb++){
                int row = ja*nb+kb;
                d[ja*nx + i][kb] = row<ny? ((double)data[row*nx + i] - mu[row])/std[row] : 0;
                // d[row*na_v + ka][kb] = (ka*nb_v +kb)<nx? ((double)data[row*nx + ka*nb_v +kb]-mu[row])/std[row] : 0;
            }
        }
    }

    // V5: using 8x8 blocks
    // #pragma omp parallel for schedule(static, 1)
    // for(int ia=0; ia<na; ia++){
    //     for(int ja=ia; ja<na; ja++){
    //         double4_t dot00={0}, dot01={0}, dot10={0}, dot11={0};
    //         for(int k=0; k<nx; k++){
    //             constexpr int PF = 10;
    //             __builtin_prefetch(&d[ia*nx + k + PF]);
    //             __builtin_prefetch(&d[ja*nx + k + PF]);
    //             double4_t a00 = d[ia*nx+k];
    //             double4_t b00 = d[ja*nx+k];
    //             double4_t a10 = swap2(a00);
    //             double4_t b01 = swap1(b00);
    //             dot00 += a00*b00;
    //             dot01 += a00*b01;
    //             dot10 += a10*b00;
    //             dot11 += a10*b01;
    //         }
    //         dot10 = swap2(dot10);
    //         dot11 = swap2(dot11);
    //         double4_t dot[4] = {dot00, dot01, dot10, dot11};
    //         // for(int tmp=0; tmp<4; tmp++){
    //         //     printf("dot[%d]: %lf, %lf, %lf, %lf\n", tmp, dot[tmp][0], dot[tmp][1], dot[tmp][2], dot[tmp][3]);
    //         // }

    //         for(int ib=0; ib<nb; ib++){
    //             for(int jb=0; jb<nb; jb++){
    //                 int i = ia*nb + ib;
    //                 int j = ja*nb + jb;
    //                 if(i<=j && i<ny && j<ny){
    //                     result[i*ny+j] = (float)dot[jb^ib][ib];
    //                 }
    //             }
    //         }
            
            
    //     }
    // }



    // V5: using 16x16 blocks now
    #pragma omp parallel for schedule(static, 1)
    for(int ia=0; ia<na-1; ia+=2){
        for(int ja=ia; ja<na-1; ja+=2){
            double4_t dot00_ab={0}, dot01_ab={0}, dot10_ab={0}, dot11_ab={0};
            double4_t dot00_ad={0}, dot01_ad={0}, dot10_ad={0}, dot11_ad={0};
            double4_t dot00_cb={0}, dot01_cb={0}, dot10_cb={0}, dot11_cb={0};
            double4_t dot00_cd={0}, dot01_cd={0}, dot10_cd={0}, dot11_cd={0};
            for(int k=0; k<nx; k++){
                // constexpr int PF = 10;
                // __builtin_prefetch(&d[ia*nx + k + PF]);
                // __builtin_prefetch(&d[ja*nx + k + PF]);
                double4_t a00 = d[ia*nx+k];
                double4_t b00 = d[ja*nx+k];
                double4_t c00 = d[(ia+1)*nx+k];
                double4_t d00 = d[(ja+1)*nx+k];
                double4_t a10 = swap2(a00);
                double4_t b01 = swap1(b00);
                double4_t c10 = swap2(c00);
                double4_t d01 = swap1(d00);
                dot00_ab += a00*b00;
                dot01_ab += a00*b01;
                dot10_ab += a10*b00;
                dot11_ab += a10*b01;
                dot00_ad += a00*d00;
                dot01_ad += a00*d01;
                dot10_ad += a10*d00;
                dot11_ad += a10*d01;
                dot00_cb += c00*b00;
                dot01_cb += c00*b01;
                dot10_cb += c10*b00;
                dot11_cb += c10*b01;
                dot00_cd += c00*d00;
                dot01_cd += c00*d01;
                dot10_cd += c10*d00;
                dot11_cd += c10*d01;
            }
            dot10_ab = swap2(dot10_ab);
            dot11_ab = swap2(dot11_ab);
            dot10_ad = swap2(dot10_ad);
            dot11_ad = swap2(dot11_ad);
            dot10_cb = swap2(dot10_cb);
            dot11_cb = swap2(dot11_cb);
            dot10_cd = swap2(dot10_cd);
            dot11_cd = swap2(dot11_cd);
            double4_t dot[4][4] = {{dot00_ab, dot01_ab, dot10_ab, dot11_ab}, {dot00_ad, dot01_ad, dot10_ad, dot11_ad}, {dot00_cb, dot01_cb, dot10_cb, dot11_cb}, {dot00_cd, dot01_cd, dot10_cd, dot11_cd}};
            // double4_t dot_ad[4] = {dot00_ad, dot01_ad, dot10_ad, dot11_ad};
            // double4_t dot_cb[4] = {dot00_cb, dot01_cb, dot10_cb, dot11_cb};
            // double4_t dot_cd[4] = {dot00_cd, dot01_cd, dot10_cd, dot11_cd};
            // for(int tmp=0; tmp<4; tmp++){
            //     printf("dot[%d]: %lf, %lf, %lf, %lf\n", tmp, dot[tmp][0], dot[tmp][1], dot[tmp][2], dot[tmp][3]);
            // }
            for(int block=0; block<4; block++){
                for(int ib=0; ib<nb; ib++){
                    for(int jb=0; jb<nb; jb++){
                        int i = (ia+block/2)*nb + ib;
                        int j = (ja+block%2)*nb + jb;
                        if(i<=j && i<ny && j<ny){
                            result[i*ny+j] = (float)dot[block][jb^ib][ib];
                        }
                    }
                }
            }
            
            
        }
    }

    // V4: using 3x3 blocks
    // #pragma omp parallel for schedule(static, 1)
    // for(int row0=0; row0<ncd-2; row0+=3){
    //     int row1 = row0+1;
    //     int row2 = row0+2;
    //     for(int row3=row0; row3<ncd-2; row3+=3){
    //         int row4 = row3+1;
    //         int row5 = row3+2;
    //         double4_t dot[nd][nd] = {0};
    //         // double4_t dot14={0}, dot15={0}, dot16={0}, dot24={0}, dot25={0}, dot26={0}, dot34={0}, dot35={0}, dot36={0};
    //         for(int k=0; k<na_v; k++){
    //             double4_t d0 = d[row0*na_v+k];
    //             double4_t d1 = d[row1*na_v+k];
    //             double4_t d2 = d[row2*na_v+k];
    //             double4_t d3 = d[row3*na_v+k];
    //             double4_t d4 = d[row4*na_v+k];
    //             double4_t d5 = d[row5*na_v+k];
    //             dot[0][0] += d0*d3;
    //             dot[0][1] += d0*d4;
    //             dot[0][2] += d0*d5;
    //             dot[1][0] += d1*d3;
    //             dot[1][1] += d1*d4;
    //             dot[1][2] += d1*d5;
    //             dot[2][0] += d2*d3;
    //             dot[2][1] += d2*d4;
    //             dot[2][2] += d2*d5;
    //         }
    //         for(int i=0; i<nd; i++){
    //             for(int j=0; j<nd; j++){
    //                 if((row0+i)<=(row3+j) and (row0+i)<ny and (row3+j)<ny){
    //                     result[(row0+i)*ny + row3+j] = (float)hsum4(dot[i][j]);
    //                 }
    //             }
    //         }
    //     }
    // }
    std::free(d);
}