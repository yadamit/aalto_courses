#include "is.h"
#include<cmath>
#include "stopwatch.h"
// #include <new>
#include<cstdlib>
#include <x86intrin.h>
// #include<iostream>
// #include <tuple>
// #include <functional>

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
    // because 4th entry is always zero
    for(int i=0; i<3; i++){
        sum += x[i];
    }
    return sum;
}

struct Best{
    double util;
    int x0, y0, x1, y1;
};

Result segment(int ny, int nx, const float* data) {
    // FIXME
    // print data
    // printf("data:\n");
    // // for(int i=0; i<ny*nx*3; i++){
    // //     printf("%f ", data[i]);
    // // }
    // printf("\n");
    // for(int j=0; j<ny; j++){
    //     for(int i=0; i<nx; i++){
    //         printf("(");
    //         for(int c=0; c<3; c++){
    //             printf("%f ", data[j*nx*3+3*i+c]);
    //         }
    //         printf(") ");
    //     }
    //     printf("\n");
    // }
    ppc::stopwatch sw;
    sw.record();

    // TODO: we aready have sum of previous enteries, so don't need to calculate all the sum
    double4_t* sum = double4_alloc((ny+1)*(nx+1));
    double4_t zero_vec = {0};
    // for all rectangles
    #pragma omp parallel for schedule(static, 1)
    for(int y=0; y<=ny; y++){
        for(int x=0; x<=nx; x++){
            if(x==0 or y==0){
                sum[y*(nx+1)+x] = zero_vec;
                continue;
            }
            // sum[y*nx+x] = {0}; //vector of 4 doubles //TODO: change it to a double4_t variable
            double rect_sum[3] = {sum[y*(nx+1)+x-1][0], sum[y*(nx+1)+x-1][1], sum[y*(nx+1)+x-1][2]}; //3 colors
            for(int j=0; j<y; j++){
                for(int c=0; c<3; c++){
                    rect_sum[c] += (double)data[j*nx*3 + (x-1)*3 + c];
                }
            }
            // double rect_sum[3] = {0};
            // for(int j=0; j<y; j++){
            //     for(int i=0; i<x; i++){
            //         for(int c=0; c<3; c++){
            //             rect_sum[c] += data[j*nx*3 + i*3 + c];
            //         }
            //     }
            // }
            sum[y*(nx+1)+x][0] = rect_sum[0];
            sum[y*(nx+1)+x][1] = rect_sum[1];
            sum[y*(nx+1)+x][2] = rect_sum[2];
            sum[y*(nx+1)+x][3] = 0;
            // printf("x,y:(%d,%d) rect_sum:(%lf, %lf, %lf, %lf)\n", x,y,sum[y*(nx+1)+x][0], sum[y*(nx+1)+x][1], sum[y*(nx+1)+x][2], sum[y*(nx+1)+x][3]);
        }
    }
    sw.record();

    // sum till i,j = sum[i+1, j+1]
    double4_t total_sum = sum[(ny)*(nx+1)+(nx)];
    // struct Best best_starts[nx*ny]={0,0,0,0,0};
    struct Best* best_starts = (struct Best*)calloc(nx*ny, sizeof(struct Best));
    #pragma omp parallel for schedule(static, 1)

    // loop on size
    for(int size=0; size<nx*ny; size++){
        int size_x = size/ny+1;
        int size_y = (size+1)%ny+1;
        // printf("size:%d, size_x:%d,, size_y:%d\n", size, size_x, size_y);
        double max_util = -std::numeric_limits<double>::infinity();
        int  best_x0=0, best_y0=0, best_x1=0, best_y1=0;
        
        int size_in = size_x*size_y;
        int size_out = nx*ny - size_in;
        if(size_out==0)continue; //TODO: check on this statement
        double reciproc_size_in = 1.0/size_in;
        double reciproc_size_out = 1.0/size_out;
        double reciproc_size_in_out = reciproc_size_out+reciproc_size_in;
        double4_t two_total_sum_size_out = 2*reciproc_size_out*total_sum;
        double hsum_total_sq_sum = reciproc_size_out*hsum4(total_sum*total_sum);
        // printf("hsum_total_sq_sum=%lf\n", hsum_total_sq_sum);
        for(int y0=0; y0<=ny-size_y; y0++){
            for(int x0=0; x0<=nx-size_x; x0++){
                // int x1 = x0+size_x-1;
                // int y1 = y0+size_y-1;
                int tmp_x1 = x0+size_x; //because we are only using x1+1
                int tmp_y1 = y0+size_y;
                int new_nx = nx+1;
                // double4_t sum_in = sum[(y1+1)*new_nx+(x1+1)] - sum[(y1+1)*new_nx+(x0)] - sum[(y0)*new_nx+(x1+1)] + sum[(y0)*new_nx+(x0)];
                double4_t sum_in = sum[tmp_y1*new_nx+tmp_x1] - sum[tmp_y1*new_nx+(x0)] - sum[(y0)*new_nx+tmp_x1] + sum[(y0)*new_nx+(x0)];
                // double4_t sum_out = total_sum - sum_in;

                double util = hsum4(reciproc_size_in_out*sum_in*sum_in - two_total_sum_size_out*sum_in);
                // printf("(%d,%d,%d,%d) size_in:%d, size_out:%d, util:%lf\n", x0,y0,x1,y1, size_in, size_out, util);

                // util = hsum4(sum_in*sum_in*reciproc_size_in + sum_out*sum_out*reciproc_size_out);   
                // printf("(%d,%d,%d,%d) size_in:%d, size_out:%d, util:%lf\n", x0,y0,x1,y1, size_in, size_out, util);
                // this util = original util * size_in*size_out
                // printf("sum_in: %lf,%lf,%lf,%lf\n", sum_in[0], sum_in[1], sum_in[2], sum_in[3]);
                // printf("sum_out: %lf,%lf,%lf,%lf\n", sum_out[0], sum_out[1], sum_out[2], sum_out[3]);
                // printf("(%d,%d,%d,%d) size_in:%d, size_out:%d, util:%lf\n", x0,y0,x1,y1, size_in, size_out, util);
                if(util>max_util){
                    max_util = util;
                    best_x0 = x0;
                    best_y0 = y0;
                    best_x1 = tmp_x1-1;;
                    best_y1 = tmp_y1-1;
                }
            }
        }
        best_starts[size].util = max_util + hsum_total_sq_sum;//+size_out*hsum4(total_sum*total_sum);
        best_starts[size].x0 = best_x0;
        best_starts[size].x1 = best_x1;
        best_starts[size].y0 = best_y0;
        best_starts[size].y1 = best_y1;

    }




    // // loop on starting point
    // for(int x0=0; x0<nx; x0++){
    //     double max_util = 0;
    //     int  best_y0=0, best_x1=0, best_y1=0;
    //     for(int y0=0; y0<ny; y0++){
    //         for(int y1=y0; y1<ny; y1++){
    //             for(int x1=x0; x1<nx; x1++){
    //                 // for each rectangle

    //                 int size_in = (x1-x0+1)*(y1-y0+1);
    //                 int size_out = nx*ny - size_in;
    //                 // if(size_out==0){
    //                 //     continue;
    //                 // }
    //                 int new_nx = nx+1;
    //                 double4_t sum_in = sum[(y1+1)*new_nx+(x1+1)] - sum[(y1+1)*new_nx+(x0)] - sum[(y0)*new_nx+(x1+1)] + sum[(y0)*new_nx+(x0)];
    //                 double4_t sum_out = total_sum - sum_in;
    //                 double util = (size_out==0)? hsum4(sum_in*sum_in)/size_in : hsum4(sum_in*sum_in)/size_in + hsum4(sum_out*sum_out)/size_out;   
    //                 // printf("sections:\n");
    //                 // printf("section3: %lf,%lf,%lf,%lf\n", sum[(y1+1)*(nx+1)+(x1+1)][0], sum[(y1+1)*(nx+1)+(x1+1)][1], sum[(y1+1)*(nx+1)+(x1+1)][2], sum[(y1+1)*(nx+1)+(x1+1)][3]);
    //                 // printf("section2: %lf,%lf,%lf,%lf\n", sum[(y1+1)*(nx+1)+(x0)][0], sum[(y1+1)*(nx+1)+(x0)][1], sum[(y1+1)*(nx+1)+(x0)][2], sum[(y1+1)*(nx+1)+(x0)][3]);
    //                 // printf("section1: %lf,%lf,%lf,%lf\n", sum[(y0)*(nx+1)+(x1+1)][0], sum[(y0)*(nx+1)+(x1+1)][1], sum[(y0)*(nx+1)+(x1+1)][2], sum[(y0)*(nx+1)+(x1+1)][3]);
    //                 // printf("section0: %lf,%lf,%lf,%lf\n", sum[(y0)*(nx+1)+(x0)][0], sum[(y0)*(nx+1)+(x0)][1], sum[(y0)*(nx+1)+(x0)][2], sum[(y0)*(nx+1)+(x0)][3]);
    //                 // printf("sum_in: %lf,%lf,%lf,%lf\n", sum_in[0], sum_in[1], sum_in[2], sum_in[3]);
    //                 // printf("sum_out: %lf,%lf,%lf,%lf\n", sum_out[0], sum_out[1], sum_out[2], sum_out[3]);
    //                 // printf("(%d,%d,%d,%d) size_in:%d, size_out:%d, util:%lf\n", x0,y0,x1,y1, size_in, size_out, util);
    //                 if(util>max_util){
    //                     max_util = util;
    //                     // best_x0 = x0;
    //                     best_y0 = y0;
    //                     best_x1 = x1;
    //                     best_y1 = y1;
    //                 }

    //             }
    //         }
    //     }
    //     best_starts[x0].util = max_util;
    //     best_starts[x0].x1 = best_x1;
    //     best_starts[x0].y0 = best_y0;
    //     best_starts[x0].y1 = best_y1;
    // }
    sw.record();
    
    double max_util = 0;
    int best_x0=0, best_x1=0, best_y0=0, best_y1=0;
    for(int size=0; size<nx*ny; size++){
        if(best_starts[size].util>max_util){
            max_util = best_starts[size].util;
            best_x0 = best_starts[size].x0;
            best_x1 = best_starts[size].x1;
            best_y0 = best_starts[size].y0;
            best_y1 = best_starts[size].y1;
        }
    }


    int x0=best_x0, x1=best_x1, y0=best_y0, y1=best_y1;
    double4_t sum_in = sum[(y1+1)*(nx+1)+(x1+1)] - sum[(y1+1)*(nx+1)+(x0)] - sum[(y0)*(nx+1)+(x1+1)] + sum[(y0)*(nx+1)+(x0)];
    double4_t sum_out = total_sum - sum_in;
    int size_in = (best_x1-best_x0+1)*(best_y1-best_y0+1);
    int size_out = nx*ny - size_in;

    Result result = {
        best_y0,
        best_x0,
        best_y1+1,
        best_x1+1,
        {(float)sum_out[0]/size_out, (float)sum_out[1]/size_out, (float)sum_out[2]/size_out},
        {(float)sum_in[0]/size_in, (float)sum_in[1]/size_in, (float)sum_in[2]/size_in}
    };
    sw.record();
    // sw.print();
    free(sum);
    free(best_starts);
    return result;
}