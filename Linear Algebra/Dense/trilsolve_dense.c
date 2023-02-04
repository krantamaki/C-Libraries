#include "declare_dense.h"


// Function for solving a system of linear equations Lx = b, where 
// L is an invertible lower triangular matrix
// Returns 0 if operation is successful 1 otherwise
int trilsolve_dense(denseMatrix* L, denseMatrix* x, denseMatrix* b) {
	// Check that the matrix dimensions match
	if (!(L->m == x->n && L->n == b->n && x->m == b->m)) {
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
	if (!(L->n == L->m)) {
		printf("\nERROR: Matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (L->proper_init || x->proper_init || b->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy L or b create copies of them
	denseMatrix* _L = alloc_denseMatrix(L->n, L->n);
	denseMatrix* _b = alloc_denseMatrix(b->n, 1);
	
	// Check that the allocation was successful
	if (_L->proper_init || _b->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
		// Even in case of error free the allocated memory
		free_denseMatrix(_L);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// Copy the contents of L into _L
	copy_dense(_L, L);
	copy_dense(_b, b);
	
	// The main loop body
	for (int i = 0; i < _L->n - 1; i++) {
		// Split the linear system into blocks and allocate memory
		// for needed subarrays
		
		// For L
		double l11;
		_apply_dense(_L, &l11, i, i);
		denseMatrix* l21 = _subarray_dense(_L, i + 1, _L->n, i, i + 1);
		// Check that l11 is not 0
		if (!(l11 != 0)) {
			printf("\nERROR: Matrix is not invertible\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_L);
			free_denseMatrix(_b);
			free_denseMatrix(l21);
			
			return 1;
		}
		
		// For b
		double b1;
		_apply_dense(_b, &b1, i, 0);
		denseMatrix* b2 = _subarray_dense(_b, i + 1, _b->n, 0, 1);
		
		// Check that the operations were successful
		if (l21->proper_init || b2->proper_init) {
			printf("\nERROR: Subarray allocation failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_L);
			free_denseMatrix(_b);
			free_denseMatrix(l21);
			free_denseMatrix(b2);
			
			return 1;
		}
		
		// Update x
		double x_i = b1 / l11;
		_place_dense(x, x_i, i, 0);
		
		// Update b
		smult_dense(l21, l21, x_i);
		diff_dense(b2, l21, b2);
		_place_subarray_dense(_b, b2, i + 1, _b->n, 0, 1);
		
		// Free temporary arrays
		free_denseMatrix(l21);
		free_denseMatrix(b2);
	}
	
	// Update the final element of x
	double l11;
	_apply_dense(_L, &l11, _L->n - 1, _L->n - 1);
	// Check that l11 is not 0
	if (!(l11 != 0)) {
		printf("\nERROR: Trilsolve failed as L is not invertible\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
		// Even in case of error free the allocated memory
		free_denseMatrix(_L);
		free_denseMatrix(_b);
			
		return 1;
	}
	
	double b1;
	_apply_dense(_b, &b1, _b->n - 1, 0);
	_place_dense(x, b1 / l11, x->n - 1, 0);
	
	// Free allocated memory
	free_denseMatrix(_L);
	free_denseMatrix(_b);
	
	return 0;
}
