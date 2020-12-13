#include "so.h"
#include <algorithm>

void swap(data_t* data, int i, int j)
{
	data_t tmp;
	tmp = data[i];
	data[i] = data[j];
	data[j] = tmp;
}

int partition(data_t* data, int start, int end){
    // partition the array based on pivot
    // choosing pivot as middle
	int mid = (start+end)/2;
	data_t pivot = data[mid];
	int ptr=start;
	swap(data, mid, end);
	for(int i=start; i<end; ++i){
		if(data[i] < pivot){
			swap(data, i, ptr);
			ptr++;
		}
	}
	swap(data, ptr, end);
	return ptr;
}

void quicksort(data_t* data, int start, int end){
	if(start>=end)
		return;
    if (end-start<50){
        std::sort(data+start, data+end+1);
        return;
    }
	int pivot = partition(data, start, end);
    // make two child tasks
	#pragma omp task
	quicksort(data, start, pivot-1);
	#pragma omp task
	quicksort(data, pivot+1, end);
}

void psort(int n, data_t* data) {
    #pragma omp parallel
    #pragma omp single
    {
        quicksort(data, 0 , n-1);
    }
}