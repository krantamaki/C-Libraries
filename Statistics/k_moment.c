#include "declare_stats.h"


// Function that computes the k-th raw sample moment
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int k_moment(denseMatrix* A, int k, double* ret) {
	// Check that matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the mean
	double mean_val;
	mean(A, &mean_val);
	int size = A->n * A->m;
	
	double sum = 0.0;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			sum += pow(val - mean_val, (double)k);
		}
	}
	
	*ret = sum / size;
}
