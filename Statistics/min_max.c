#include "declare_stats.h"


// Function that finds the largest element in the matrix
// Uses a brute force search
// Returns 0 if operation is successful and 1 otherwise
int max(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	double best_found = -DBL_MAX;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			if (val > best_found) best_found = val;
		}
	}
	
	// Store the found value
	*ret = best_found;
}


// Function that finds the smallest element in the matrix
// Uses a brute force search
// Returns 0 if operation is successful and 1 otherwise
int min(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	double best_found = DBL_MAX;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			if (val < best_found) best_found = val;
		}
	}
	
	// Store the found value
	*ret = best_found;
}
