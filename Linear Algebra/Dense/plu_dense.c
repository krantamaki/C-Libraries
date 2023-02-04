#include "declare_dense.h"


// Function for generic PLU decomposition. Works for invertible (nonsingular) matrices
// NOTE! The passed arguments P, L and U should all be initialized to 0
// Returns 0 if operation is successful 1 otherwise
int PLU_dense(denseMatrix* A, denseMatrix* P, denseMatrix* L, denseMatrix* U) {
	// Check that the matrix dimensions match
	if (!(A->n == L->n && A->m == L->m && A->n == U->n && A->m == U->m && A->n == P->n && A->m == P->m)) {
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
	// Check that the matrices are large enough
	if (!(A->n > 1)) {
		printf("\nERROR: Matrix isn't large enough\n");
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
	
	// Generate an identity matrix to help with computations
	denseMatrix* P2 = eye_dense(A->n, A->n);
	
	// Allocate other helpers
	denseMatrix* a21 = eye_dense(A->n, 1);
	double a11 = (double)0.0;
	
	// Check that the allocations were successful
	if (_A->proper_init || P2->proper_init || a21->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		free_denseMatrix(P2);
		free_denseMatrix(a21);
		
		return 1;
	}
	
	// Copy the contents of A into _A
	copy_dense(_A, A);
	
	// Turn the permutation matrix P into an identity matrix
	init_eye_dense(P);

	
	// Compute the Frobenius norm of _A
	double4_t frob_vect = {(double)0.0, (double)0.0, (double)0.0, (double)0.0};
	int vect_num = _A->vects_per_row;
	// Go over the rows
	for (int n = 0; n < _A->n; n++) {
		// Go over vectors per row
		for (int vect = 0; vect < vect_num; vect++) {
			frob_vect += _A->data[vect_num * n + vect] * _A->data[vect_num * n + vect];
		}
	}
	double frob = sqrt(frob_vect[0] + frob_vect[1] + frob_vect[2] + frob_vect[3]);
	
	// Define the tolerance with the Frobenius norm and machine epsilon
	double tol = frob * DBL_EPSILON;
	
	// Main loop body
	for (int k = 0; k < _A->n - 1; k++) {
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
		// matrix be singular and thus won't have a PLU decomposition
		if (pivot < tol) {
			printf("\nERROR: Given matrix is singular\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_A);
			free_denseMatrix(P2);
			free_denseMatrix(a21);
			
			return 1;
		}
		
		// If the best pivot is not found on row k swap the rows in P2
		if (pivot_i != k) {
			#pragma omp parallel for schedule(dynamic, 1)
			for (int vect = 0; vect < vect_num; vect++) {
				double4_t tmp = P2->data[vect_num * k + vect];
				P2->data[vect_num * k + vect] = P2->data[vect_num * pivot_i + vect];
				P2->data[vect_num * pivot_i + vect] = tmp;
			}
		}
		
		// Allocate memory for temporary matrices
		denseMatrix* PA = alloc_denseMatrix(_A->n, _A->n);
		denseMatrix* P2_T = alloc_denseMatrix(_A->n, _A->n);
		denseMatrix* A22_tmp = alloc_denseMatrix(_A->n - (k + 1), _A->n - (k + 1));
		double a11_2 = 0;
		// Check that the allocations were successful
		if (PA->proper_init || P2_T->proper_init || A22_tmp->proper_init) {
			printf("\nERROR: Memory allocation for a temporary matrix failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
			// Even in case of error free the allocated memory
			free_denseMatrix(PA);
			free_denseMatrix(P2_T);
			free_denseMatrix(A22_tmp);
			free_denseMatrix(_A);
			free_denseMatrix(P2);
			free_denseMatrix(a21);
			
			return 1;
		}
		
		// Get the needed subarrays
		transpose_dense(P2, P2_T);
		mult_dense(P2_T, _A, PA);
		_apply_dense(PA, &a11_2, k, k);
		denseMatrix* a21_2 = _subarray_dense(PA, k + 1, _A->n, k, k + 1);
		denseMatrix* a21_tmp = _subarray_dense(a21, k, _A->n, 0, 1);
		denseMatrix* a12 = _subarray_dense(PA, k, k + 1, k + 1, _A->n);
		denseMatrix* A22 = _subarray_dense(PA, k + 1, _A->n, k + 1, _A->n);
		denseMatrix* P22 = _subarray_dense(P, k, _A->n, k, _A->n);
		denseMatrix* P2_tmp = _subarray_dense(P2, k, _A->n, k, _A->n);
		denseMatrix* P2_tmp_T = _subarray_dense(P2_T, k, _A->n, k, _A->n);
		// Check that the operations were successful
		if (a21_2->proper_init || a21_tmp->proper_init || a12->proper_init || A22->proper_init || P22->proper_init || P2_tmp->proper_init || P2_tmp_T->proper_init) {
			printf("\nERROR: Subarray allocation failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
			// Even in case of error free the allocated memory
			free_denseMatrix(PA);
			free_denseMatrix(P2_T);
			free_denseMatrix(A22_tmp);
			free_denseMatrix(a21_2);
			free_denseMatrix(a21_tmp);
			free_denseMatrix(a12);
			free_denseMatrix(A22);
			free_denseMatrix(P22);
			free_denseMatrix(P2_tmp);
			free_denseMatrix(P2_tmp_T);
			free_denseMatrix(_A);
			free_denseMatrix(P2);
			free_denseMatrix(a21);
			
			return 1;
		}
		
		// Start filling up the matrices
		_place_dense(L, (double)1.0, k, k);
		_place_dense(U, a11_2, k, k);
		_place_subarray_dense(U, a12, k, k + 1, k + 1, _A->n);
		
		// Update _A
		mult_dense(a21_2, a12, A22_tmp);
		smult_dense(A22_tmp, A22_tmp, 1.0 / a11_2);
		diff_dense(A22, A22_tmp, A22);
		_place_subarray_dense(_A, A22, k + 1, _A->n, k + 1, _A->n);
		
		// Update P and L
		if (k > 0) {
			mult_dense(P22, P2_tmp, P22);
			_place_subarray_dense(P, P22, k, _A->n, k, _A->n);
			mult_dense(P2_tmp_T, a21_tmp, a21_tmp);
			smult_dense(a21_tmp, a21_tmp, 1.0 / a11);
			_place_subarray_dense(L, a21_tmp, k, _A->n, k - 1, k); 
		} 
		
		// Update a21 and a11
		_place_subarray_dense(a21, a21_2, k + 1, _A->n, 0, 1);
		a11 = a11_2;
		
		// Free temporary arrays
		free_denseMatrix(PA);
		free_denseMatrix(P2_T);
		free_denseMatrix(A22_tmp);
		free_denseMatrix(a21_2);
		free_denseMatrix(a21_tmp);
		free_denseMatrix(a12);
		free_denseMatrix(A22);
		free_denseMatrix(P22);
		free_denseMatrix(P2_tmp);
		free_denseMatrix(P2_tmp_T);
	}
	
	// Add base case to the matrices
	// For U
	double A_nn;
	_apply_dense(_A, &A_nn, _A->n - 1, _A->n - 1);
	_place_dense(U, A_nn, _A->n - 1, _A->n - 1);
	// For L
	_place_dense(L, (double)1.0, _A->n - 1, _A->n - 1);
	double P2_nn;
	_apply_dense(P2, &P2_nn, _A->n - 1, _A->n - 1);
	double a21_n;
	_apply_dense(a21, &a21_n, _A->n - 1, 0);
	_place_dense(L, P2_nn * a21_n / a11, _A->n - 1, _A->n - 2);	
	
	// Free rest of the arrays
	free_denseMatrix(_A);
	free_denseMatrix(P2);
	free_denseMatrix(a21);
	
	return 0;
}
