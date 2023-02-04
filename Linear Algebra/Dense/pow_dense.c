#include "declare_dense.h"


// (Naive) Function for matrix powers
// Returns 0 if operation is successful 1 otherwise
int pow_dense(denseMatrix* A, denseMatrix* ret, const int k) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is symmetric
	if (!(A->n == A->m)) {
		printf("\nERROR: Matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Create an identity matrix of correct size which will used to compute the power
	denseMatrix* I = eye_dense(A->n, A->n);
	// Check that the operation was successful
	if (I->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(I);
		
		return 1;
	}
	
	// Allocate memory for a temporary ret matrix. This allows
	// calling this function in form pow_dense(A, B, A)
	denseMatrix* tmp = alloc_denseMatrix(ret->n, ret->m);
	// Check that allocation was successful
	if (tmp->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(I);
		free_denseMatrix(tmp);
		
		return 1;
	}
	 
	// Multiply A with itself k times in a loop
	for (int i = 0; i < k; i++) {
		mult_dense(A, I, tmp);
		copy_dense(I, tmp);
	}
	
	// Copy the contents of tmp to ret
	copy_dense(ret, tmp);

	// Free the temporary matrix
	free_denseMatrix(I);
	free_denseMatrix(tmp);
	
	return 0;
}  
