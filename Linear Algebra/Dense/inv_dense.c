#include "declare_dense.h"


// Function for inverting a matrix using Gauss-Jordan method
// Modified from: https://rosettacode.org/wiki/Gauss-Jordan_matrix_inversion#C
// Returns 0 if operation is successful 1 otherwise
int inv_dense(denseMatrix* A, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
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
	
	// Create an identity matrix of correct size which will converted to the inverse
	denseMatrix* I = eye_dense(A->n, A->n);
	// Check that the operation was successful
	if (I->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
									
		// Even in case of error free the allocated memory
		free_denseMatrix(I);
		
		return 1;
	}
	
	// As we don't want to destroy A create a copy of it
	denseMatrix* _A = alloc_denseMatrix(A->n, A->n);
	if (_A->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
											
		// Even in case of error free the allocated memory
		free_denseMatrix(I);
		free_denseMatrix(_A);
		
		return 1;
	}
	
	// Copy the contents of A into _A
	copy_dense(_A, A);
	
	// Compute the Frobenius norm of _A
	double4_t frob_vect = {(double)0.0, (double)0.0, (double)0.0, (double)0.0};
	int vect_num = _A->vects_per_row;
	// Go over the rows
	for (int i = 0; i < _A->n; i++) {
		// Go over vectors per row
		for (int vect = 0; vect < vect_num; vect++) {
			frob_vect += _A->data[vect_num * i + vect] * _A->data[vect_num * i + vect];
		}
	}
	double frob = sqrt(frob_vect[0] + frob_vect[1] + frob_vect[2] + frob_vect[3]);
	
	// Define the tolerance with the Frobenius norm and machine epsilon
	double tol = frob * DBL_EPSILON;
	
	// Main loop body
	for (int k = 0; k < _A->n; k++) {
		// Find the pivot
		int vect0 = k / DOUBLE_ELEMS;
		int elem0 = k % DOUBLE_ELEMS;
		double pivot = fabs(_A->data[vect_num * k + vect0][elem0]);
		int pivot_i = k;
		
		// Go over the remaining rows and check if there is a greater pivot
		#pragma omp parallel for schedule(dynamic, 1)
		for (int i = k + 1; i < _A->n; i++) {
			double opt_pivot = fabs(_A->data[vect_num * i + vect0][elem0]);
			if (opt_pivot > pivot) {
				pivot = opt_pivot;
				pivot_i = i;
			}
		}
		
		// If the found pivot is smaller than the given tolerance must the
		// matrix be singular and thus won't have a inverse
		if (pivot < tol) {
			printf("\nERROR: Given matrix is singular\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
														
			// Even in case of error free the allocated memory
			free_denseMatrix(I);
			free_denseMatrix(_A);
		
			return 1;;
		}
		
		// If the best pivot is not found on row k swap the rows in both 
		// matrix _A aswell as in the matrix I
		if (pivot_i != k) {
			// Swap rows in _A
			#pragma omp parallel for schedule(dynamic, 1)
			for (int vect = vect0; vect < vect_num; vect++) {
				double4_t tmp = _A->data[vect_num * k + vect];
				_A->data[vect_num * k + vect] = _A->data[vect_num * pivot_i + vect];
				_A->data[vect_num * pivot_i + vect] = tmp;
			}
			// Swap rows in I
			#pragma omp parallel for schedule(dynamic, 1)
			for (int vect = 0; vect < vect_num; vect++) {
				double4_t tmp = I->data[vect_num * k + vect];
				I->data[vect_num * k + vect] = I->data[vect_num * pivot_i + vect];
				I->data[vect_num * pivot_i + vect] = tmp;
			}
		}
		
		// Scale the rows of both _A and I so that the pivot is 1
		pivot = (double)1.0 / pivot;
		double4_t pivot_vect = {pivot, pivot, pivot, pivot};
		// Scale _A
		#pragma omp parallel for schedule(dynamic, 1)
		for (int vect = vect0; vect < vect_num; vect++) {
			_A->data[vect_num * k + vect] *= pivot_vect;
		}
		// Scale I
		#pragma omp parallel for schedule(dynamic, 1)
		for (int vect = 0; vect < vect_num; vect++) {
			I->data[vect_num * k + vect] *= pivot_vect;
		}
		
		// Subtract to get the wanted zeros in _A
		// Go over the rows
		#pragma omp parallel for schedule(dynamic, 1)
		for (int i = 0; i < _A->n; i++) {
			if (i == k) continue;
			double row_pivot = _A->data[i * vect_num + vect0][elem0];
			double4_t row_pivot_vect = {row_pivot, row_pivot, row_pivot, row_pivot};
			// Go over each vector on the row for _A
			for (int vect = vect0; vect < vect_num; vect++) {
				_A->data[vect_num * i + vect] -= _A->data[vect_num * k + vect] * row_pivot_vect;
			}
			// Go over each vector on the row for I
			for (int vect = 0; vect < vect_num; vect++) {
				I->data[vect_num * i + vect] -= I->data[vect_num * k + vect] * row_pivot_vect;
			}		
		}	
	}
	
	// Copy the final inverse from I to ret
	copy_dense(ret, I);
	
	// Free the allocated temporary memory
	free_denseMatrix(I);
	free_denseMatrix(_A);
	
	return 0;
}
