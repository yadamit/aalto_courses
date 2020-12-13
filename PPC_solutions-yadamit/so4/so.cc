#include "so.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
using namespace std;

inline void merge(data_t *a, int size, int *idx, int num_threads){
  int tmp_size = size - size%num_threads;
  while (num_threads>1){
    int thread_len = tmp_size/num_threads;
    #pragma omp parallel for schedule(static, 1)
    for(int i=0; i<num_threads; i++ )
      idx[i]=i*thread_len;
    idx[num_threads] = size;

    #pragma omp parallel for schedule(static, 1)
    for(int i=0; i<num_threads; i+=2 ){
      inplace_merge(a+idx[i],a+idx[i+1],a + idx[i+2]);
    }
    num_threads=num_threads>>1;
  }

}
 
void psort(int n, data_t* data){
  int num_threads = omp_get_max_threads();
  unsigned int v = num_threads;
  // printf("Num_threads:%d\n", num_threads);
  // printf("v:%d\n", v);
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  // if(num_threads!=(int)v) num_threads = (int)(v>>1);
  // num_threads = (num_threads==(int)v)? num_threads : (int)(v>>1);
  num_threads = (int)v;
  // printf("Num_threads:%d\n", num_threads);


  // if(num_threads>8) num_threads=8;
  // else if(num_threads==5 or num_threads==3) num_threads--;
  // else if(num_threads==6 or num_threads==7) num_threads=4;

  int* idx = (int*)malloc((num_threads+1)*sizeof(int));
  int thread_len = n/num_threads;
  #pragma omp parallel for schedule(static, 1)
  for(int i=0; i<num_threads; i++){
    idx[i] = i*thread_len;
  }
  idx[num_threads] = n;
  
  #pragma omp parallel for num_threads(num_threads)
  for(int t=0; t<num_threads; t++){
    std::sort(data+idx[t], data+idx[t+1]);
  }
  merge(data, n, idx, num_threads);
  free(idx);
}