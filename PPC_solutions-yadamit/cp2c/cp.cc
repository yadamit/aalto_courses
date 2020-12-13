#include "cp.h"
#include<cmath>
#include <new>
#include<cstdlib>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

double hsum4(double4_t x){
    double sum = 0;
    for(int i=0; i<4; i++){
        sum += x[i];
    }
    return sum;
}

void correlate(int ny, int nx, const float* data, float* result) {
    // TODO
    const int nb = 4;
    int na = (nx-1)/nb + 1;
    int nab = na*nb;
    double4_t* d = double4_alloc(ny*na);

    double mu[ny] = {0};
    double std[ny] = {0};
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
    for(int row=0; row<ny; row++){
        for(int ka=0; ka<na; ka++){
            for(int kb=0; kb<nb; kb++){
                d[row*na + ka][kb] = (ka*nb +kb)<nx? ((double)data[row*nx + ka*nb +kb]-mu[row])/std[row] : 0;
            }
        }
    }


    for(int row2=0; row2<ny; row2++){
        for(int row1=row2; row1<ny; row1++){
            // double dot = 0;
            double4_t dot = {0};
            for(int ka=0; ka<na; ka++){
                dot = dot + (d[row1*na + ka]*d[row2*na+ka]);
                // dot += (double)data[row1*nx+i]*(double)data[row2*nx+i];
            }
            // dot -= nx*mu[row1]*mu[row2];
            // result[row1 + row2*ny] = (float)corr(row1, row2, data, nx, mu[row1], mu[row2], std[row1], std[row2]);
            result[row1 + row2*ny] = (float)hsum4(dot);
        }
    }
    std::free(d);
}