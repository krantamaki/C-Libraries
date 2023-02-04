#include "declare_dense.h"


// Function for computing the Hadamard product (element-wise product A.*B)
// of two denseMatrices
// Returns 0 if operation is successful 1 otherwise
int hprod_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == B->n && A->n == ret->n && A->m == B->m && A->m == ret->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
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
			ret->data[vect_num * i + vect] = A->data[vect_num * i + vect] * B->data[vect_num * i + vect];
		}
	}
	
	return 0; 
}
