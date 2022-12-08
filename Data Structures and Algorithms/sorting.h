#ifndef SORTING
#define SORTING

static const int THRESHOLD = 128;

// Declare insertion sort
void _insertion_sort(void *arr, const int start, const int end, 
					 size_t size, int (*cmp)(void*, void*));
					 
void insertion_sort(void *arr, const int n, size_t size,
					int (*cmp)(void*, void*));

// Declare mergesort				
void _merge(void* arr, const int start, const int mid, const int end,
			size_t size, int (*cmp)(void*, void*));
			
void _mergesort(void* arr, const int start, const int end, 
				size_t size, int (*cmp)(void*, void*));
				
void mergesort(void* arr, const int n, size_t size,
			   int (*cmp)(void*, void*));
			   
// Declare quicksort
intTuple _partition_3way(void* arr, const int start, const int end, 
					 size_t size, int (*cmp)(void*, void*));
					 
void _quicksort(void* arr, const int start, const int end, size_t size,
				int (*cmp)(void*, void*));
				
void quicksort(void* arr, const int n, size_t size,
			   int (*cmp)(void*, void*));
			   
// Declare radix sort 
void int_radix_sort(void* arr, const int n, size_t size,
			        int (*cmp)(void*, void*));

// Declare testing functions        
int validate_sort(void* arr, const int n, size_t size,
				  int (*cmp)(void*, void*));

double sort_timer(void (*sort)(void *, const int, size_t, int (*cmp)(void*, void*)),
				  void* arr, const int n, size_t size, int (*cmp)(void*, void*)); 

// Declare main function (for testing purposes)		  
//int main();

#endif
