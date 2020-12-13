#include "mf.h"
#include<algorithm>
// #include <functional>


void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    // FIXME
    // let me fix you
    for(int y=0; y<ny; y++){
        for(int x=0; x<nx; x++){
            int lw, rw, uh, bh; //left width, right width, upper height, below height
            lw = (x>hx)? hx : x;
            rw = ((nx-x)>hx)? hx : (nx-x-1);
            uh = (y>hy)? hy : y;
            bh = ((ny-y-1)>hy)? hy : (ny-y-1);
            int size = (lw+rw+1)*(uh+bh+1);
            float arr[size] = {0};
            int counter=0;
            for(int j=y-uh; j<=y+bh; j++){
                for(int i=x-lw; i<=x+rw; i++){
                    arr[counter] = in[i + j*nx];
                    counter++;
                }
            }
            std::nth_element(arr, arr+size/2, arr+size);
            out[x + y*nx] = arr[size/2];
            if(size%2==0){
                std::nth_element(arr, arr+size/2-1, arr+size);
                out[x + y*nx] += arr[size/2-1];
                out[x + y*nx] /= 2;
            }
        }
    }

    // you feel okay now?
}
