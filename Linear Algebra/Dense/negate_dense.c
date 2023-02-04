#include "declare_dense.h"


// Function for negating a matrix
// Returns 0 if operation is successful, 1 otherwise
int negate_dense(denseMatrix* A, denseMatrix* ret) {
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
	smult_dense(A, ret, (double)-1.0);
	
	return 0;
}
