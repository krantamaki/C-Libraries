#include "declare_stats.h"


// Finds the b quantile of values found in the matrix
// Uses quickselect algorithm in the search
// Returns 0 if operation is successful and 1 otherwise
int b_quantile(denseMatrix* A, double b, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the given b is valid
	if (!(0.0 < b < 1.0)) {
		printf("\nERROR: Given b not in needed range 0 < b < 1\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Turn the denseMatrix into a double array
	double* data = conv_to_arr(A);
	int n = A->m * A->n;
	int b_i = (int)b * n;
	
	quickselect(data, n, b_i, sizeof(double), double_cmp_approx);
	*ret = data[b_i];
}
