#ifndef SORTING
#define SORTING

static const int THRESHOLD = 64;

int int_cmp(const void* ptr1, const void* ptr2, const int desc);

int float_cmp(const void* ptr1, const void* ptr2, const int desc);

int double_cmp(const void* ptr1, const void* ptr2, const int desc);

int str_cmp(const void* ptr1, const void* ptr2, const int desc);

void _place(void* src, void* dst, size_t size);

void* _apply(void* arr, const int i, size_t size);

void _insertion_sort(void *arr, const int start, const int end, 
					 const int desc, size_t size, 
					 int (*cmp)(const void *, const void *, const int));
					 
void insertion_sort(void *arr, const int n, const int desc, size_t size,
					int (*cmp)(const void *, const void *, const int));
					
void _merge(void* arr, const int start, const int mid, const int end,
			const int desc, size_t size, 
			int (*cmp)(const void *, const void *, const int));
			
void _mergesort(void* arr, const int start, const int end, 
				const int desc, size_t size,
				int (*cmp)(const void *, const void *, const int));
				
void mergesort(void* arr, const int n, const int desc, size_t size,
			   int (*cmp)(const void *, const void *, const int));
			   
void _swap(void* elem1, void* elem2, size_t size);

// Struct to hold the return value of the _partition function
typedef struct {
	int _1;
	int _2;
} int_tuple;

int_tuple _partition(void* arr, const int start, const int end, 
					 const int desc, size_t size,
					 int (*cmp)(const void *, const void *, const int));
					 
void _quicksort(void* arr, const int start, const int end, 
				const int desc, size_t size,
				int (*cmp)(const void *, const void *, const int));
				
void quicksort(void* arr, const int n, const int desc, size_t size,
			   int (*cmp)(const void *, const void *, const int));
			   
void int_radix_sort(void* arr, const int n, const int desc, size_t size,
			        int (*cmp)(const void *, const void *, const int));
			        
int validate_sort(const void* arr, const int n, const int desc, size_t size,
				  int (*cmp)(const void *, const void *, const int));
				  
void rand_int_arr(int* arr, const int n, const int upper);

void rand_float_arr(float* arr, const int n, const float upper);

double sort_timer(void (*sort)(void *, const int, const int, size_t, int*),
				  void* arr, const int n, const int desc, size_t size,
				  int (*cmp)(const void *, const void *, const int));
				  
int main();


#endif
