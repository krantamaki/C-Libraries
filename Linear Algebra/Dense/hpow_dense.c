#include "declare_dense.h"


// Function for computing the Hadamard power (element-wise power)
// of a given matrix
// Returns 0 if operation is successful 1 otherwise
int hpow_dense(denseMatrix* A, denseMatrix* ret, double k) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
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
	
	int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			double4_t tmp = A->data[vect_num * i + vect];
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				ret->data[vect_num * i + vect][elem] = pow(tmp[elem], k);
			}
		}
	}
	
	return 0; 
}
