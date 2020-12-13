#include "cp.h"
#include<cmath>

double corr(int row1, int row2, const float* data, int nx, double mu1, double mu2, double std1, double std2){
    // ignore this function, it's never used.
    double dot = 0;
    for(int i=0; i<nx; i++){
        double tmp1, tmp2;
        tmp1 = (double)data[row1*nx+i];
        tmp2 = (double)data[row2*nx+i];
        dot += tmp1*tmp2;
    }
    dot = dot - nx*mu1*mu2;
    return dot/(std1*std2);
}

void correlate(int ny, int nx, const float* data, float* result) {
    // TODO
    double mu[ny] = {0};
    double std[ny] = {0};
    
    const int nb0 = 10;
    int na0 = (nx-1)/nb0 + 1;
    int nab0 = na0*nb0;
    
    for(int row=0; row<ny; row++){
        double tmp_mu[nb0] = {0};
        double tmp_std[nb0] = {0};
        for(int j=0; j<na0-1; j++){
            for(int k=0; k<nb0; k++){
                double tmp;
                tmp = (double)data[row*nx+nb0*j+k];
                tmp_mu[k] += tmp;
                tmp_std[k] += tmp*tmp;
            }
        }
        for(int i=nab0-nb0; i<nx; i++){
            double tmp;
            tmp = (double)data[row*nx+i];
            tmp_mu[0] += tmp;
            tmp_std[0] += tmp*tmp;
        }
        double tmp_mu_tot = 0;
        double tmp_std_tot = 0;
        for(int i=0; i<nb0; i++){
            tmp_mu_tot += tmp_mu[i];
            tmp_std_tot += tmp_std[i];
        }
        mu[row] = tmp_mu_tot/nx;
        std[row] = (tmp_std_tot - nx*mu[row]*mu[row]);
    }


    const int nb = 25;
    int na = (nx-1)/nb + 1;
    int nab = na*nb;
    for(int row2=0; row2<ny; row2++){
        for(int row1=row2; row1<ny; row1++){
            double dot[nb] = {0};
            for(int i=0; i<na-1; i++){
                for(int j=0; j<nb; j++){
                    dot[j] += (double)data[row1*nx+nb*i+j]*(double)data[row2*nx+nb*i+j];
                }
                // dot1 += (double)data[row1*nx+3*i]*(double)data[row2*nx+3*i];
                // dot2 += (double)data[row1*nx+3*i+1]*(double)data[row2*nx+3*i+1];
                // dot3 += (double)data[row1*nx+3*i+2]*(double)data[row2*nx+3*i+2];
            }
            for(int i=nab-nb; i<nx; i++){
                dot[0] += (double)data[row1*nx+i]*(double)data[row2*nx+i];
            }
            double dot1 = 0;
            for(int i=0; i<nb; i++){
                dot1 += dot[i];
            }
            dot1 -= nx*mu[row1]*mu[row2];
            // result[row1 + row2*ny] = (float)corr(row1, row2, data, nx, mu[row1], mu[row2], std[row1], std[row2]);
            result[row1 + row2*ny] = dot1/sqrt(std[row1]*std[row2]);
        }
    }
}