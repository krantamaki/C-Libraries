#include "declare_stats.h"


// Function for computing the median abslute deviation for given data.
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int mad(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Allocate memory for a secondary matrix to store the elementwise
	// absolute deviations
	denseMatrix* _A = alloc_denseMatrix(A->n, A->m);
	if (_A->proper_init) {
		printf("\nERROR: Memory allocation for temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		free_denseMatrix(_A);
		return 1;
	}
	
	// Compute the mean
	double mean_val;
	mean(A, %mean_val);
	
	int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < _A->n; i++) {
		// Go over the columns
		for (int j = 0; j < _A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			double abs_dev = fabs(val - mean);
			_place_dense(_A, abs_dev, i, j);
		}
	}
	
	// Find the median of the absolute deviations
	median(_A, &ret);
	
	free_denseMatrix(_A);
}
