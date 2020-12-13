#include "cp.h"
#include<cmath>

void correlate(int ny, int nx, const float* data, float* result) {
    // TODO
    double mu[ny] = {0};
    double std[ny] = {0};
    #pragma omp parallel for
    for(int row=0; row<ny; row++){
        for(int j=0; j<nx; j++){
            double tmp;
            tmp = (double)data[row*nx+j];
            mu[row] += tmp;
            std[row] += tmp*tmp;
        }
        mu[row] /= nx;
        std[row] = (std[row] - nx*mu[row]*mu[row]);
    }
    #pragma omp parallel for schedule(static, 1)
    for(int row2=0; row2<ny; row2++){
        for(int row1=row2; row1<ny; row1++){
            double dot = 0;
            for(int i=0; i<nx; i++){
                dot += (double)data[row1*nx+i]*(double)data[row2*nx+i];
            }
            dot -= nx*mu[row1]*mu[row2];
            // result[row1 + row2*ny] = (float)corr(row1, row2, data, nx, mu[row1], mu[row2], std[row1], std[row2]);
            result[row1 + row2*ny] = dot/sqrt(std[row1]*std[row2]);
        }
    }
}