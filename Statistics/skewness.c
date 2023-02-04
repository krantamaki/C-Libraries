#include "declare_stats.h"


// Function that computes the sample skewness for given data
// Computed across all elements independent of the shape of the matrix
// If skewness factor > 0 the distribution is skewed to the right
// and if skewness factor < 0 the distribution is skewed to the left
// Returns 0 if operation is successful and 1 otherwise
int skewness(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the 3rd moment
	double moment_3;
	k_moment(A, 3, &moment_3);
	
	// Compute the standard deviation
	double sd_val;
	sd(A, &sd_val);
	
	*ret = moment_3 / pow(sd_val, 3.0);
}
