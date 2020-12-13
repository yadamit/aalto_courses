#include "cp.h"
#include<cmath>

double corr(int row1, int row2, const float* data, int nx, double mu1, double mu2, double std1, double std2){
    double dot = 0;
    // double mag1 = 0;
    // double mag2 = 0;
    // double mu1 = 0;
    // double mu2 = 0;
    // for(int i=0; i<nx; i++){
    //     mu1 += data[row1*nx+i];
    //     mu2 += data[row2*nx+i];
    // }
    // mu1 /= nx;
    // mu2 /= nx;
    for(int i=0; i<nx; i++){
        double tmp1, tmp2;
        tmp1 = (double)data[row1*nx+i];
        tmp2 = (double)data[row2*nx+i];
        dot += tmp1*tmp2;
        // mag1 += tmp1*tmp1;
        // mag2 += tmp2*tmp2;
        // mu1 += tmp1;
        // mu2 += tmp2;
    }
    // mu1 /= nx;
    // mu2 /= nx;
    dot = dot - nx*mu1*mu2;
    // mag1 = mag1 - nx*mu1*mu1;
    // mag2 = mag2 - nx*mu2*mu2;
    // return dot/(sqrt(mag1*mag2));
    return dot/(std1*std2);

    
}

void correlate(int ny, int nx, const float* data, float* result) {
    // TODO
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
        std[row] = (std[row] - nx*mu[row]*mu[row]);
    }
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