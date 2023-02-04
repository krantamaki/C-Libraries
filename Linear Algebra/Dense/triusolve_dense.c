#include "declare_dense.h"


// Function for solving a system of linear equations Ux = b, where 
// U is an invertible upper triangular matrix
// Returns 0 if operation is successful 1 otherwise
int triusolve_dense(denseMatrix* U, denseMatrix* x, denseMatrix* b) {
	// Check that the matrix dimensions match
	if (!(U->m == x->n && U->n == b->n && x->m == b->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that c and b are vectors
	if (!(x->m == 1)) {
		printf("\nERROR: Passed arguments x and b have to be column vectors\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is symmetric
	if (!(U->n == U->m)) {
		printf("\nERROR: Matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (U->proper_init || x->proper_init || b->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy L or b create copies of them
	denseMatrix* _U = alloc_denseMatrix(U->n, U->n);
	denseMatrix* _b = alloc_denseMatrix(b->n, 1);
	
	// Check that the allocation was successful
	if (_U->proper_init || _b->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
		// Even in case of error free the allocated memory
		free_denseMatrix(_U);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// Copy the contents of U into _U and b into _b
	copy_dense(_U, U);
	copy_dense(_b, b);
	
	// The main loop body
	for (int i = _U->n - 1; i > 0; i--) {
		// Split the linear system into blocks and allocate memory
		// for needed subarrays
		
		// For U
		double u22;
		_apply_dense(_U, &u22, i, i);
		// Check that u22 is not 0
		if (!(u22 != 0)) {
			printf("\nERROR: Matrix is not invertible\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_U);
			free_denseMatrix(_b);
			
			return 1;
		}	

		denseMatrix* u12 = _subarray_dense(_U, 0, i, i, i + 1);
		
		// For b
		double b2;
		_apply_dense(_b, &b2, i, 0);
		denseMatrix* b1 = _subarray_dense(_b, 0, i, 0, 1);
		
		// Check that the operations were successful
		if (u12->proper_init || b1->proper_init) {
			printf("\nERROR: Subarray allocation failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_U);
			free_denseMatrix(_b);
			free_denseMatrix(u12);
			free_denseMatrix(b1);
			
			return 1;
		}
		
		// Update x
		double x_i = b2 / u22;
		_place_dense(x, x_i, i, 0);
		
		// Update b
		smult_dense(u12, u12, x_i);
		diff_dense(b1, u12, b1);
		_place_subarray_dense(_b, b1, 0, i, 0, 1);
		
		// Free temporary arrays
		free_denseMatrix(u12);
		free_denseMatrix(b1);
	}

	// Handle the final element of x
	double u22;
	_apply_dense(_U, &u22, 0, 0);
	// Check that u22 is not 0
	if (!(u22 != 0)) {
		printf("\nERROR: Matrix is not invertible\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_U);
		free_denseMatrix(_b);
			
		return 1;
	}
	double b2;
	_apply_dense(_b, &b2, 0, 0);
	_place_dense(x, b2 / u22, 0, 0);

	// Free allocated memory
	free_denseMatrix(_U);
	free_denseMatrix(_b);
	
	return 0;
}
