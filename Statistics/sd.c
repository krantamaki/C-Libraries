#include "declare_stats.h"


// Function for computing the sample standard deviation for given data.
// Computes the standard deviation across all elements independent of the 
// shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int sd(denseMatrix* A, double* ret) {
	// Check that vector is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	double variance;
	var(A, &variance);
	*ret = pow(variance, 1/2);
}
