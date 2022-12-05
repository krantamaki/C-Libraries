#ifndef SORTING
#define SORTING

static const int THRESHOLD = 128;

typedef struct {
	int _1;
	int _2;
} int_tuple;

// Declare comparison functions
int int_cmp(void* ptr1, void* ptr2);

int float_cmp(void* ptr1, void* ptr2);

int double_cmp(void* ptr1, void* ptr2);

int str_cmp(void* ptr1, void* ptr2);

// Declare void array helpers
void* _apply(void* arr, const int i, size_t size);

void _place(void* dst, void* src, size_t size);

void _swap(void* elem1, void* elem2, size_t size);

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
int_tuple _partition(void* arr, const int start, const int end, 
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
				  
void rand_int_arr(int* arr, const int n, const int upper);

void rand_float_arr(float* arr, const int n, const float upper);

double sort_timer(void (*sort)(void *, const int, size_t, int (*cmp)(void*, void*)),
				  void* arr, const int n, size_t size, int (*cmp)(void*, void*)); 

// Declare main function (for testing purposes)		  
//int main();


#endif
