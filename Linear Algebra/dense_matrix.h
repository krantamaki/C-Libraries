#ifndef DENSE_MATRIX
#define DENSE_MATRIX

typedef struct {
	int n;
	int m;
	int proper_init;  // 0 if proper, 1 otherwise
	size_t vects_per_row;
	double4_t* data;
} denseMatrix


#endif
