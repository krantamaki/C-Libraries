#include "declare_stats.h"


// Function for computing the weighted mean for given data.
// Computes the weighted mean across all elements independent of the shape
// or the matrix.
// Assumes that the given weights are proper i.e. 0 < W_ij < 1 and 
// the weights sum to 1.
// Returns 0 if operation is successful and 1 otherwise 
int weighted_mean(denseMatrix* A, denseMatrix* W, double* ret) {
	// Check that the dimensions of the matrices match
	if (!(A->n == W->n && A->m == W->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are properly allocated
	if (A->proper_init || W->proper_init) {
		printf("\nERROR: Matrices not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int size = A->n * A->m;
	double sum = 0.0;
	const int vect_num = A->vects_per_row;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		double4_t sum_vect = {0.0, 0.0, 0.0, 0.0};
		// Go over the columns
		for (int vect = 0; vect < vect_num; vect++) {
			sum_vect += W->data[vect_num * i + vect] * A->data[vect_num * i + vect];
		}
		sum += sum_vect[0] + sum_vect[1] + sum_vect[2] + sum_vect[3];
	}
	
	// Store the found mean
	*ret = sum / size;
}
