#ifndef GENERAL
#define GENERAL

typedef struct {
	int _1;
	int _2;
} intTuple;

// Define a vector that can hold 8 floats (for 256 bit wide vector registers)
static const int FLOAT_ELEMS = 8;
typedef float float8_t __attribute__ ((vector_size (FLOAT_ELEMS * sizeof(float))));

// Define a vector that can hold 4 doubles (for 256 bit wide vector registers)
static const int DOUBLE_ELEMS = 4;
typedef double double4_t __attribute__ ((vector_size (DOUBLE_ELEMS * sizeof(double))));

// Declare void array helpers
void* _apply(void* arr, const int i, size_t size);

void _place(void* dst, void* src, size_t size);

void _swap(void* elem1, void* elem2, size_t size);

// Declare bit operations
int get_bit(char byte, int i);

char toggle_bit(char byte, int i);

// Declare comparison functions
int int_cmp(void* ptr1, void* ptr2);

int float_cmp(void* ptr1, void* ptr2);

int double_cmp(void* ptr1, void* ptr2);

int double_cmp_approx(void* ptr1, void* ptr2);

int str_cmp(void* ptr1, void* ptr2);

// Declare math functions
int _pow(int x, int n);

int _ceil(int a, int b);

// Declare array creation functions
void rand_int_arr(int* arr, const int n, const int upper);

void rand_float_arr(float* arr, const int n, const float upper);

#endif
