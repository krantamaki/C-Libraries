#include "declare_dense.h"


// Function for transposing a given matrix
// Returns 0 if operation is successful 1 otherwise
int transpose_dense(denseMatrix* A, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->m && A->m == ret->n)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}

	// Go over the rows of A
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			// Get value at index i,j from A and place it at j,i in ret
			double val;
			_apply_dense(A, &val, i, j);
			_place_dense(ret, val, j, i);
		}
	}
	
	return 0;
}
