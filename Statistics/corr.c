#include "declare_stats.h"


// Function that computes the sample correlation of the values in two matrices
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int corr(denseMatrix* A, denseMatrix* B, double* ret) {
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
	
	// Compute the standard deviations for the matrices
	double sd_A, sd_B;
	sd(A, &sd_A);
	sd(B, &sd_B);
	
	// Compute the covariance between the matrices
	double cov_AB;
	cov(A, B, &cov_AB);
	
	*ret = cov_AB / (sd_A * sd_B);
}
