#ifndef DENSE_MATRIX
#define DENSE_MATRIX

typedef struct {
	int n;  // Number of rows
	int m;  // Number of columns
	int proper_init;  // 0 if proper, 1 otherwise
	size_t vects_per_row;
	double4_t* data;
} denseMatrix;


// DECLARE GENERAL FUNCTIONS

// Function for printing a matrix
int MAX_PRINTS = 5;  // Maximum number of rows printed and values per row printed
int print_dense(denseMatrix* A);

// Functions for allocating and freeing matrices
denseMatrix* alloc_denseMatrix(int n, int m);

denseMatrix* conv_to_denseMatrix(double* arr, int n, int m);

denseMatrix* eye_dense(int n, int m);

void free_denseMatrix(denseMatrix* A);

// Functions for slicing matrices
int _apply_dense(denseMatrix* A, double* ret, int i, int j);

int _place_dense(denseMatrix* A, double val, int i, int j);

denseMatrix* _subarray_dense(denseMatrix* A, int n_start, int n_end, int m_start, int m_end);

int _place_subarray_dense(denseMatrix* A, denseMatrix* B, int n_start, int n_end, int m_start, int m_end);

// Misc
int init_eye_dense(denseMatrix* ret);

int copy_dense(denseMatrix* dst, denseMatrix* src);



#endif
