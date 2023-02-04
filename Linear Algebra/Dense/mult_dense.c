#include "declare_dense.h"


// Function for multiplying two matrices (i.e AB)
// Multiplies th matrices in a naive way
// Returns 0 if operation is successful 1 otherwise
int mult_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->m == B->n && A->n == ret->n && B->m == ret->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// For linear reading we need the transpose of B
	// so allocate memory for this transose
	denseMatrix* B_T = alloc_denseMatrix(B->m, B->n);
	// Check that allocation was successful
	if (B_T->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(B_T);
		
		return 1;
	}
	// and transpose B
	transpose_dense(B, B_T);
	
	// Allocate memory for a temporary ret matrix. This allows
	// calling this function in form mult_dense(A, B, A)
	denseMatrix* tmp = alloc_denseMatrix(ret->n, ret->m);
	// Check that allocation was successful
	if (tmp->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(B_T);
		free_denseMatrix(tmp);
		
		return 1;
	}
	
	
	const int vect_num = A->vects_per_row;
	// Go over the rows of A
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the rows of B_T
		for (int j = 0; j < B_T->n; j++) {
			// Multiply the values for the rows element-wise and sum them together
			double4_t sum = {(double)0.0, (double)0.0, (double)0.0, (double)0.0};
			for (int vect = 0; vect < vect_num; vect++) {
				sum += A->data[vect_num * i + vect] * B_T->data[vect_num * j + vect];
			}

			double val = 0.0;
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				val += sum[elem];	
			}
			
			// Place the value at the correct location
			 _place_dense(tmp, val, i, j);
		}
	}
	
	// Copy the contents of tmp to ret
	copy_dense(ret, tmp);
	
	// Free allocated memory
	free_denseMatrix(B_T);
	free_denseMatrix(tmp);
	
	return 0;
}
