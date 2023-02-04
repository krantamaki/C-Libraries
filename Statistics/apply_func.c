#include "declare_stats.h"


// Function for applying a given function to each row of a matrix.
// Function must take as parameters a pointer to a single row of data
// and a pointer to a double value and return an int.
// NOTE! The ret matrix should be a column vector of proper size
// Returns 0 if operation is successful and 1 otherwise
int apply_rowfunc(denseMatrix* A, denseMatrix* ret, 
				  int (*func)(denseMatrix*, double*)) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && ret->m == 1)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR: Matrices not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Get the corresponding row
		denseMatrix* row = _subarray_dense(A, i, i + 1, 0, A->m);
		
		// Check that allocation was successful
		if (row->proper_init) {
			printf("\nERROR: Matrix not properly allocated\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}
		
		// Apply the wanted function on the row
		double tmp;
		if ((*func)(row, &tmp)) {
			printf("\nERROR: Function application failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}
		
		free_denseMatrix(row);
		_apply_dense(ret, &tmp, i, 0);
	}  
}


// Function for applying a given function to each column of a matrix.
// Function must take as parameters a pointer to a single column of data
// and a pointer to a double value and return an int.
// NOTE! The ret matrix should be a column vector of proper size
// Returns 0 if operation is successful and 1 otherwise
int apply_colfunc(denseMatrix* A, denseMatrix* ret, 
				  int (*func)(denseMatrix*, double*)) {
	// Check that the matrix dimensions match
	if (!(A->m == ret->n && ret->m == 1)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR: Matrices not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Go over the columns
	#pragma omp parallel for schedule(dynamic, 1)
	for (int j = 0; j < A->m; i++) {
		// Get the corresponding row
		denseMatrix* col = _subarray_dense(A, 0, A->n, j, j + 1);
		
		// Check that allocation was successful
		if (col->proper_init) {
			printf("\nERROR: Matrix not properly allocated\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}
		
		// Apply the wanted function on the row
		double tmp;
		if ((*func)(col, &tmp)) {
			printf("\nERROR: Function application failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}
		
		free_denseMatrix(col);
		_apply_dense(ret, &tmp, j, 0);
	}  
}
