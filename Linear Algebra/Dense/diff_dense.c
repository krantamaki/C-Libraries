#include "declare_dense.h"


// Function for taking the difference between two matrices (i.e. A - B))
// Returns 0 if operation is successful, 1 otherwise
int diff_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == B->n && A->n == ret->n && A->m == B->m && A->m == ret->m)) {
		printf("\nERROR CODE 2: Difference failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Difference failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			ret->data[vect_num * i + vect] = A->data[vect_num * i + vect] - B->data[vect_num * i + vect];
		}
	}
	
	return 0;
}
