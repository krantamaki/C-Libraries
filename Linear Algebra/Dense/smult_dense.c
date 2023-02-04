#include "declare_dense.h"


// Function for multiplying a denseMatrix with a scalar
// Returns 0 if operation is successful 1 otherwise
int smult_dense(denseMatrix* A, denseMatrix* ret, const double c) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that the matrices is properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const double4_t mlpr = {c, c, c, c};
	const int vect_num = A->vects_per_row;
	// Go over the rows of A
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over each vector on the row
		for (int vect = 0; vect < vect_num; vect++) {
			ret->data[vect_num * i + vect] = A->data[vect_num * i + vect] * mlpr;
		}
	}
	
	return 0;
}
