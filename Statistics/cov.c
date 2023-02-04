#include "declare_stats.h"


// Function that computes the sample covariance of the values in two matrices
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int cov(denseMatrix* A, denseMatrix* B, double* ret) {
	// Check that the dimensions of the matrices match
	if (!(A->n == B->n && A->m == B->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are properly allocated
	if (A->proper_init || B->proper_init) {
		printf("\nERROR: Matrices not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the means
	double mean_A, mean_B;
	mean(A, &mean_A);
	mean(B, &mean_B);
	
	int size = A->n * A->m;
	double sum = 0.0;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double a_ij, b_ij;
			_apply_dense(A, &a_ij, i, j);
			_apply_dense(B, &b_ij, i, j);
			sum += (a_ij - mean_A) * (b_ij - mean_B);
		}
	}
	
	*ret = sum / (size - 1);
}
