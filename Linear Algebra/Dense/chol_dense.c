#include "declare_dense.h"


// Function for Cholensky decomposition A = LL^T. Works only for s.p.d matrices
// Returns 0 if operation is successful 1 otherwise
int chol_dense(denseMatrix* A, denseMatrix* L) {
	// Check that the matrix dimensions match
	if (!(A->n == L->n && A->m == L->m)) {
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
	if (A->proper_init || L->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy A create a copy of it
	denseMatrix* _A = alloc_denseMatrix(A->n, A->n);
	if (_A->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
	
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		
		return 1;
	}
	
	// Copy the contents of A into _A
	copy_dense(_A, A);
	
	// Go over the rows of A
	for (int i = 0; i < _A->n - 1; i++) {
		// Allocate memory for update values for L
		denseMatrix* a21_T = alloc_denseMatrix(1, _A->n - (i + 1));
		denseMatrix* A22_tmp = alloc_denseMatrix(_A->n - (i + 1), _A->n - (i + 1));
		
		// Get the needed subarrays
		double a11;
		_apply_dense(_A, &a11, i, i);
		if (a11 <= 0) {
			printf("\nERROR: Matrix is not symmetric positive definite\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
			// Even in case of error free the allocated memory
			free_denseMatrix(_A);
			free_denseMatrix(a21_T);
			free_denseMatrix(A22_tmp);
		
			return 1;
		}
		
		denseMatrix* a21 = _subarray_dense(_A, i + 1, _A->n, i, i + 1);
		denseMatrix* A22 = _subarray_dense(_A, i + 1, _A->n, i + 1, _A->n);
		// Check that the allocations were successful
		if (A22->proper_init || a21->proper_init || A22_tmp->proper_init || a21_T->proper_init) {
			printf("\nERROR: Subarray allocation failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
							
			// Even in case of error free the allocated memory
			free_denseMatrix(_A);
			free_denseMatrix(a21_T);
			free_denseMatrix(A22_tmp);
			free_denseMatrix(a21);
			free_denseMatrix(A22);
		
			return 1;
		}

		// Compute the update values for L and A
		transpose_dense(a21, a21_T);
		mult_dense(a21, a21_T, A22_tmp);
		smult_dense(A22_tmp, A22_tmp, 1. / a11);
		diff_dense(A22, A22_tmp, A22);
		
		double l11 = sqrt(a11);
		smult_dense(a21, a21, 1. / l11);
		
		// Place the values back to the arrays
		_place_dense(L, l11, i, i);
		_place_subarray_dense(L, a21, i + 1, _A->n, i, i + 1);
		_place_subarray_dense(_A, A22, i + 1, _A->n, i + 1, _A->n);
		
		// Free the temporary arrays;
		free_denseMatrix(a21);
		free_denseMatrix(a21_T);
		free_denseMatrix(A22);
		free_denseMatrix(A22_tmp);
	}
	
	// Place the last value into the array
	double a11;
	_apply_dense(_A, &a11, L->n - 1, L->n - 1);
	if (a11 <= 0) {
		printf("\nERROR: Matrix is not symmetric positive definite\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		
		return 1;
	}
	
	_place_dense(L, sqrt(a11), L->n - 1, L->n - 1);
	
	// Free the copy of A
	free_denseMatrix(_A);
	
	return 0;
}
