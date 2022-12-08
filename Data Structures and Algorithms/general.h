#ifndef GENERAL
#define GENERAL

typedef struct {
	int _1;
	int _2;
} intTuple;

// Declare void array helpers
void* _apply(void* arr, const int i, size_t size);

void _place(void* dst, void* src, size_t size);

void _swap(void* elem1, void* elem2, size_t size);

// Declare comparison functions
int int_cmp(void* ptr1, void* ptr2);

int float_cmp(void* ptr1, void* ptr2);

int double_cmp(void* ptr1, void* ptr2);

int double_cmp_approx(void* ptr1, void* ptr2);

int str_cmp(void* ptr1, void* ptr2);

// Declare array creation functions
void rand_int_arr(int* arr, const int n, const int upper);

void rand_float_arr(float* arr, const int n, const float upper);

#endif
