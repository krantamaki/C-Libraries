#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../general.h"
#include "dense_matrix.h"


// (UNDER CONSTRUCTION!)
// This is a general linear algebra library using dense matrices 
// with double precision floating point numbers.
// Optimized to allow multithreading, vectorization (256 bit SIMD) and ILP, 
// but lacks prefetching, improved cache management and proper register reuse

// Unless otherwise mentioned the algorithms are based on the material
// from the course MS-E1651 Numerical Matrix Computation notes by Antti 
// Hannukainen or from the book Numerical Linear Algebra by Trefethen and Bau

// ERROR CODES:
// Most of the functions return an integer value which signifies either failure
// or success of the function and a reason for the possible failure. 
// - 0: Success
// - 1: Failure due to improper initialization of a matrix
// - 2: Failure due to mismatching matrix dimensions
// - 3: Failure due to improper indexing
// - 4: Failure due to matrix not being symmetric
// - 5: Failure due to matrix not being a column vector
// - 6: Failure due to matrix not being invertible (generic)
// - 7: Failure due to matrix not being invertible (singular)
// - 8: Failure due to matrix not being invertible (not s.p.d)


// GENERAL FUNCTIONS

// Function for printing a matrix. The elements will always be printed
// with precision of 3 decimal points. The 
int print_dense(denseMatrix* A) {
	// Check that matrices are properly allocated
	if (A->proper_init) {
		printf("\nERROR CODE 1: Printing failed as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int vect_num = A->vects_per_row;
	printf("\n");
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		int col_count = 0;
		// Go over the vectors in each row
		for (int vect = 0; vect < vect_num; vect++) {
			// Go over the elements in each
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				printf("%.3f\t", A->data[vect_num * i + vect][elem]);
				col_count++;
				if (col_count == MAX_PRINTS || col_count >= A->m) {
					break;
				}
			}
			if (col_count == MAX_PRINTS || col_count >= A->m) {
				col_count = 0;
				break;
			}
			
		}
		if (i == MAX_PRINTS) {
			printf("\n.\n.\n.\n");
			break;
		}
		else {
			printf("\n");
		}
	}
	
	return 0;
}


// Function for printing a complete matrix including buffer elements
// For debugging purposes only
void _print_all(denseMatrix* A) {
	printf("\n");
	for (int i = 0; i < A->n; i++) {
		for (int vect = 0; vect < A->vects_per_row; vect++) {
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				printf("%f\t", A->data[A->vects_per_row * i + vect][elem]);
			}
			printf("\n");
		}
	}
}


// Function for allocating memory for wanted sized denseMatrix
denseMatrix* alloc_denseMatrix(const int n, const int m) {
	// Check that given dimensions are positive
	if (!(n > 0 && m > 0)) {
		printf("\nERROR: Matrix dimesions must be positive\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix* ret = calloc(1, sizeof(denseMatrix));
		ret->n = n;
		ret->m = m;
		ret->vects_per_row = _ceil(m, DOUBLE_ELEMS);
		ret->data = NULL;
		ret->proper_init = 1;
		return ret;
	}
	
	// Number of vectors per row
	const int vect_num = _ceil(m, DOUBLE_ELEMS);
	// Total number of vectors
	const size_t len = n * vect_num;
	
	// Allocate aligned memory
	void* tmp = 0;
	if (posix_memalign(&tmp, sizeof(double4_t), len * sizeof(double4_t))) {
		printf("\nERROR: Aligned memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix* ret = calloc(1, sizeof(denseMatrix));
		ret->n = n;
		ret->m = m;
		ret->vects_per_row = vect_num;
		ret->data = NULL;
		ret->proper_init = 1;
		return ret;
	}
	
	// Initialize the data values as zeros
	double4_t* data = (double4_t*)tmp;
	const double4_t zeros = {(double)0.0, (double)0.0, (double)0.0, (double)0.0};
	// Go over rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < n; i++) {
		// Go over the vectors on each row
		for (int vect = 0; vect < vect_num; vect++) {
			data[vect_num * i + vect] = zeros;
		}
	}
	
	// Fill the matrix
	denseMatrix* ret = calloc(1, sizeof(denseMatrix));
	ret->n = n;
	ret->m = m;
	ret->vects_per_row = vect_num;
	ret->data = (double4_t*)tmp;
	ret->proper_init = 0;
	
	return ret;
}


// Function for generating a wanted sized identity matrix
denseMatrix* eye_dense(const int n, const int m) {
	// Check that given dimensions are positive
	if (!(n > 0 && m > 0)) {
		printf("\nERROR: Matrix dimesions must be positive\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix* ret = calloc(1, sizeof(denseMatrix));
		ret->n = n;
		ret->m = m;
		ret->vects_per_row = _ceil(m, DOUBLE_ELEMS);;
		ret->data = NULL;
		ret->proper_init = 1;
		return ret;
	}
	
	denseMatrix* ret = alloc_denseMatrix(n, m);
	// Check that allocation was successful
	if (ret->proper_init) {
		printf("\nERROR: Indentity matrix cannot be created as memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	const int lower_dim = n >= m ? m : n;
	int error_flag = 0;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < lower_dim; i++) {
		if (_place_dense(ret, (double)1.0, i, i)) {
			error_flag = 1;
		}
	}
	
	// Move error handling out of the OpenMP structured block
	if (error_flag) {
		printf("\nERROR: Indentity matrix cannot be created as there was a failure in placing a one on the diagonal\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	return ret;
}


// Function for converting a double array to denseMatrix
// Takes the first n * m elements from double array so it is assumed
// that len(arr) >= n * m
denseMatrix* conv_to_denseMatrix(double* arr, const int n, const int m) {
	// Check that given dimensions are positive
	if (!(n > 0 && m > 0)) {
		printf("\nERROR: Matrix dimesions must be positive\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix* ret = calloc(1, sizeof(denseMatrix));
		ret->n = n;
		ret->m = m;
		ret->vects_per_row = _ceil(m, DOUBLE_ELEMS);;
		ret->data = NULL;
		ret->proper_init = 1;
		return ret;
	}
	
	// Allocate memory for the denseMatrix
	denseMatrix* ret = alloc_denseMatrix(n, m);
	// Check that allocation was succesful
	if (ret->proper_init) {
		printf("\nERROR: Cannot convert to denseMatrix as memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	const int vect_num = ret->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			// Go over the elements in each vector
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				int j = vect * DOUBLE_ELEMS + elem;
				ret->data[vect_num * i + vect][elem] = i < n && j < m ? arr[i * m + j] : 0.0;
			}
		}
	}
	
	return ret;
}


// Function for freeing the memory allocated for a denseMatrix
void free_denseMatrix(denseMatrix* A) {
	// Free the data array
	free(A->data);
	// Free the struct itself
	free(A);
}


// Function for getting an individual value from a denseMatrix
int _apply_dense(denseMatrix* A, double* ret, const int i, const int j) {
	// Check that the indexes are within proper range
	if (!(i >= 0 && i < A->n && j >= 0 && j < A->m)) {
		printf("\nERROR CODE 3: Given indeces exceed the dimensions of the matrix\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	// Check that A is properly initialized
	if (A->proper_init) {
		printf("\nERROR CODE 1: Couldn't access value as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Find the proper vector and element in said vector for column j
	const int vect = j / DOUBLE_ELEMS;  // Integer division defaults to floor
	const int elem = j % DOUBLE_ELEMS;
	*ret = A->data[A->vects_per_row * i + vect][elem];
	
	return 0;
}


// Function for placing an individual value into a wanted place in a denseMatrix
int _place_dense(denseMatrix* A, const double val, const int i, const int j) {
	// Check that the wanted index is viable for the matrix
	if (!(i < A->n && j < A->m)) {
		printf("\nERROR CODE 3: Given indeces exceed the dimensions of the matrix\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR CODE 1: Couldn't place the value as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Find the proper vector and element in said vector for column j
	const int vect = j / DOUBLE_ELEMS;  // Integer division defaults to floor
	const int elem = j % DOUBLE_ELEMS;
	A->data[A->vects_per_row * i + vect][elem] = val;
	
	return 0;
}


// Function for getting an subarray of an existing denseMatrix
denseMatrix* _subarray_dense(denseMatrix* A, const int n_start, const int n_end, const int m_start, const int m_end) {
	// Check that the dimensions are proper
	if (!(n_start < n_end && n_end <= A->n && m_start < m_end && m_end <= A->m)) {
		printf("\nERROR: The start index of a slice must be smaller than end index and end index must be equal or smaller than matrix dimension\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix* ret = calloc(1, sizeof(denseMatrix));
		ret->n = n_end - n_start;
		ret->m = m_end - m_start;
		ret->vects_per_row = _ceil(ret->m, DOUBLE_ELEMS);;
		ret->data = NULL;
		ret->proper_init = 1;
		return ret;
	}
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Couldn't retrieve a subarray as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix* ret = calloc(1, sizeof(denseMatrix));
		ret->n = n_end - n_start;
		ret->m = m_end - m_start;
		ret->vects_per_row = _ceil(ret->m, DOUBLE_ELEMS);;
		ret->data = NULL;
		ret->proper_init = 1;
		return ret;
	}
	
	denseMatrix* ret = alloc_denseMatrix(n_end - n_start, m_end - m_start);
	
	// Check that allocation was successful
	if (ret->proper_init) {
		printf("\nERROR: Subarray cannot be retrieved as memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	// Go over the rows
	int error_flag = 0;
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i0 = 0; i0 < n_end - n_start; i0++) {
		int i = i0 + n_start;
		// Go over the columns
		for (int j0 = 0; j0 < m_end - m_start; j0++) {
			int j = j0 + m_start;
			double val;
			int a_success = _apply_dense(A, &val, i, j);
			int p_success = _place_dense(ret, val, i0, j0);
			
			if (a_success || p_success) {
				error_flag = 1;
			}
		}
	}
	
	// Move error handling out of the OpenMP structured block
	if (error_flag) {
		printf("\nERROR: Failed to retrieve a value\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	return ret;
}


// (Naive) Function for placing an denseMatrix B into a wanted position in another denseMatrix A
int _place_subarray_dense(denseMatrix* A, denseMatrix* B, const int n_start, const int n_end, const int m_start, const int m_end) {
	// Check that the matrix is properly allocated
	if (A->proper_init || B->proper_init) {
		printf("\nERROR CODE 1: Couldn't place the subarray as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the dimensions are proper for A
	if (!(n_start < n_end && n_end <= A->n && m_start < m_end && m_end <= A->m)) {
		printf("\nERROR CODE 3: The start index of a slice must be smaller than end index and end index must be equal or smaller than matrix dimension");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	// Check that the dimensinos are proper for B
	if (!(n_end - n_start == B->n && m_end - m_start == B->m)) {
		printf("\nERROR CODE 3: Size of the slice must correspond with the dimensions of the placed subarray\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	
	// Go over the row values
	int error_flag = 0;
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i0 = 0; i0 < n_end - n_start; i0++) {
		int i = i0 + n_start;
		// Go over the column values
		for (int j0 = 0; j0 < m_end - m_start; j0++) {
			int j = j0 + m_start;
			double B_ij;
			int a_success = _apply_dense(B, &B_ij, i0, j0);
			int p_success = _place_dense(A, B_ij, i, j);
			// Check that operations were successful
			if (a_success || p_success) {
				error_flag = a_success;
			}
		}
	}
	
	// Move error handling out of the OpenMP structured block
	if (error_flag) {
		printf("\nERROR CODE %d: Couldn't place a value from one matrix to another\n", error_flag);
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return error_flag;
	}
	
	return 0;
}


// Function for initializing an allocated denseMatrix as an identity matrix
// NOTE: Only adds the ones on the diagonal.
int init_eye_dense(denseMatrix* ret) {
	// Check that matrices are properly allocated
	if (ret->proper_init) {
		printf("\nERROR CODE 1: Initialization failed as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int lower_dim = ret->n >= ret->m ? ret->m : ret->n;
	// Go over the rows
	int error_flag = 0;
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < lower_dim; i++) {
		int p = _place_dense(ret, (double)1.0, i, i);
		if (p) {
			error_flag = p;
		}
	}
	
	// Move error handling out of the OpenMP structured block
	if (error_flag) {
		printf("\nERROR CODE %d: Indentity matrix cannot be created as there was a failure in placing a one on the diagonal\n", error_flag);
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return error_flag;
	}
	
	return 0;
}


// Function for copying the values of one matrix (src) into another (dst)
// NOTE! Could be changed to memcpy implementation (although that would 
// probably be less eficient)
int copy_dense(denseMatrix* dst, denseMatrix* src) {
	// Check that the matrix dimensions match
	if (!(dst->n == src->n && dst->m == src->m)) {
		printf("\nERROR CODE 2: Copying failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (dst->proper_init || src->proper_init) {
		printf("\nERROR CODE 1: Copying failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int vect_num = dst->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < dst->n; i++) {
		// Go over each vector on the row
		for (int vect = 0; vect < vect_num; vect++) {
			dst->data[vect_num * i + vect] = src->data[vect_num * i + vect];
		}
	}
	
	return 0;
}



// BASIC MATH OPERATIONS

// Function for summing two matrices
// Returns 0 if operation is successful, 1 otherwise
int sum_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == B->n && A->n == ret->n && A->m == B->m && A->m == ret->m)) {
		printf("\nERROR CODE 2: Summation failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Summation failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			ret->data[vect_num * i + vect] = A->data[vect_num * i + vect] + B->data[vect_num * i + vect];
		}
	}
	
	return 0;
}


// Function for taking the difference between two matrices (i.e. A - B))
// Returns 0 if operation is successful, 1 otherwise
int diff_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == B->n && A->n == ret->n && A->m == B->m && A->m == ret->m)) {
		printf("\nERROR CODE 2: Difference failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Difference failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			ret->data[vect_num * i + vect] = A->data[vect_num * i + vect] - B->data[vect_num * i + vect];
		}
	}
	
	return 0;
}


// Function for multiplying a denseMatrix with a scalar
// Returns 0 if operation is successful 1 otherwise
int smult_dense(denseMatrix* A, denseMatrix* ret, const double c) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR CODE 2: Multiplication failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that the matrices is properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Multiplication failed as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const double4_t mlpr = {c, c, c, c};
	const int vect_num = A->vects_per_row;
	// Go over the rows of A
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over each vector on the row
		for (int vect = 0; vect < vect_num; vect++) {
			ret->data[vect_num * i + vect] = A->data[vect_num * i + vect] * mlpr;
		}
	}
	
	return 0;
}


// Function for negating a matrix
// Returns 0 if operation is successful, 1 otherwise
int negate_dense(denseMatrix* A, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR CODE 2: Negation failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Negation failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	int s = smult_dense(A, ret, (double)-1.0);
	if (s) {
		printf("\nERROR CODE %d: Negation failed as an error occured with scalar multiplication\n", s);
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return s;
	}	
	
	return 0;
}


// Function for transposing a given matrix
// Returns 0 if operation is successful 1 otherwise
int transpose_dense(denseMatrix* A, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->m && A->m == ret->n)) {
		printf("\nERROR CODE 2: Transpose failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Transpose failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}

	// Go over the rows of A
	int error_flag = 0;
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			// Get value at index i,j from A and place it at j,i in ret
			double val;
			int a_success = _apply_dense(A, &val, i, j);
			int p_success = _place_dense(ret, val, j, i);
			if (a_success || p_success) {
				error_flag = a_success;
			}
		}
	}
	
	// Move error handling out of the OpenMP structured block
	if (error_flag) {
		printf("\nERROR CODE %d: Transpose failed as placing values from one matrix to another failed\n", error_flag);
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return error_flag;
	}
	
	return 0;
}


// Function for computing the Hadamard product (element-wise product A.*B)
// of two denseMatrices
// Returns 0 if operation is successful 1 otherwise
int hprod_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == B->n && A->n == ret->n && A->m == B->m && A->m == ret->m)) {
		printf("\nERROR CODE 2: Element-wise product failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Element-wise product failed as some matrix isn't properly allocated\n");
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


// Function for computing the Hadamard division (element-wise division A./B)
// of two denseMatrices. NOTE! Does not raise an error when dividing with zero,
// but leads to nan values in the matrix.
// Returns 0 if operation is successful 1 otherwise
int hdiv_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == B->n && A->n == ret->n && A->m == B->m && A->m == ret->m)) {
		printf("\nERROR CODE 2: Element-wise division failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
		printf("\nERROR: Element-wise division failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the full vectors for each row
		for (int vect = 0; vect < vect_num - 1; vect++) {
			ret->data[vect_num * i + vect] = A->data[vect_num * i + vect] / B->data[vect_num * i + vect];
		}
		// Go over the final vector element-wise (avoids division by zero)
		int max_elem = A->m % DOUBLE_ELEMS;
		for (int elem = 0; elem < max_elem; elem++) {
			double val = A->data[vect_num * i + vect_num - 1][elem] / B->data[vect_num * i + vect_num - 1][elem];
			ret->data[vect_num * i + vect_num - 1][elem] = val;
		}
	}
	
	return 0; 
}


// Function for computing the Hadamard power (element-wise power)
// of a given matrix
// Returns 0 if operation is successful 1 otherwise
int hpow_dense(denseMatrix* A, denseMatrix* ret, double k) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR CODE 2: Element-wise power failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Element-wise power failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			double4_t tmp = A->data[vect_num * i + vect];
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				ret->data[vect_num * i + vect][elem] = pow(tmp[elem], k);
			}
		}
	}
	
	return 0; 
}


// Function for computing the dot product between two column vectors
// Returns 0 if operation is successful 1 otherwise
int dot_dense(denseMatrix* v, denseMatrix* u, double* ret) {
	// Check that the matrix dimensions match
	if (!(u->m == v->m && u->n == v->n)) {
		printf("\nERROR CODE 2: Dot product failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that the 'matrices' are column vectors
	if (!(v->m == 1)) {
		printf("\nERROR CODE 5: Dot product failed as denseMatrices are not column vectors\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 5;
	}
	// Check that matrices are properly allocated
	if (v->proper_init || u->proper_init) {
		printf("\nERROR CODE 1: Dot product failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	double sum = 0.0;
	// Go over the rows
	for (int i = 0; i < u->n; i++) {
		sum += u->data[i][0] * v->data[i][0];
	}
	
	// Store the found sum
	*ret = sum;
	
	return 0;
}


// Function for multiplying two matrices (i.e AB)
// Returns 0 if operation is successful 1 otherwise
int mult_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->m == B->n && A->n == ret->n && B->m == ret->m)) {
		printf("\nERROR CODE 2: Multiplication failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || B->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Multiplication failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// For linear reading we need the transpose of B
	// so allocate memory for this transose
	denseMatrix* B_T = alloc_denseMatrix(B->m, B->n);
	// Check that allocation was successful
	if (B_T->proper_init) {
		printf("\nERROR CODE 1: Memory allocation for the transpose of B failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(B_T);
		
		return 1;
	}
	// and transpose B
	int T = transpose_dense(B, B_T);
	if (T) {
		printf("\nERROR CODE %d: Failed to transpose B\n", T);
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(B_T);
		
		return T;
	}
	
	// Allocate memory for a temporary ret matrix. This allows
	// calling this function in form mult_dense(A, B, A)
	denseMatrix* tmp = alloc_denseMatrix(ret->n, ret->m);
	// Check that allocation was successful
	if (tmp->proper_init) {
		printf("\nERROR CODE 1: Memory allocation for a temporary ret matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(B_T);
		free_denseMatrix(tmp);
		
		return 1;
	}
	
	
	const int vect_num = A->vects_per_row;
	// Go over the rows of A
	int error_flag = 0;
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
			int p = _place_dense(tmp, val, i, j);
			if (p) {
				error_flag = p;
			}
		}
	}
	
	// Move error handling out of the OpenMP structured block
	if (error_flag) {
		printf("\nERROR CODE %d: Failed to place the computed value in the correct location\n", error_flag);
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(B_T);
		free_denseMatrix(tmp);
		
		return error_flag;
	}	
	
	// Copy the contents of tmp to ret
	int c = copy_dense(ret, tmp);
	if (c) {
		printf("\nERROR CODE %d: Failed to copy the values from temporary matrix to ret\n", c);
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(B_T);
		free_denseMatrix(tmp);
		
		return c;
	}
	
	// Free allocated memory
	free_denseMatrix(B_T);
	free_denseMatrix(tmp);
	
	return 0;
}


// (Naive) Function for matrix powers
int pow_dense(denseMatrix* A, denseMatrix* ret, const int k) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR CODE 2: Matrix power failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that the matrix is symmetric
	if (!(A->n == A->m)) {
		printf("\nERROR CODE 4: Matrix power failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 4;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR CODE 1: Matrix power failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Create an identity matrix of correct size which will used to compute the power
	denseMatrix* I = eye_dense(A->n, A->n);
	// Check that the operation was successful
	if (I->proper_init) {
		printf("\nERROR CODE 1: Matrix power failed as generating an identity matrix failed\n");
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
		printf("\nERROR CODE 1: Memory allocation for a temporary ret matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(I);
		free_denseMatrix(tmp);
		
		return 1;
	}
	 
	// Multiply A with itself k times in a loop
	for (int i = 0; i < k; i++) {
		int m = mult_dense(A, I, tmp);
		if (m) {
			printf("\nERROR CODE %d: Matrix power failed as matrix multiplication failed\n", m);
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
			// In case of error free the allocated memory
			free_denseMatrix(I);
			free_denseMatrix(tmp);
		
			return m;
		}
		
		int c = copy_dense(I, tmp);
		// Copy ret value back to I
		if (c) {
			printf("\nERROR CODE %d: Matrix power failed as copying the contents from ret failed\n", c);
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
			// In case of error free the allocated memory
			free_denseMatrix(I);
			free_denseMatrix(tmp);
			
			return c;
		}
	}
	
	// Copy the contents of tmp to ret
	if (copy_dense(ret, tmp)) {
		printf("\nERROR: Failed to copy the values from temporary matrix to ret\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// In case of error free the allocated memory
		free_denseMatrix(I);
		free_denseMatrix(tmp);
		
		return 1;
	}
	
	// Free the temporary matrix
	free_denseMatrix(I);
	free_denseMatrix(tmp);
	
	return 0;
}  


// TODO: FINISH FUNCTION FOR THE TRACE

// Function for finding the trace of a square matrix (i.e. the sum
// of the diagonal elements)
//int trace_dense(denseMatrix* A, double* ret) {}


// TODO: FINISH FUNCTION FOR FINDING THE ROW ECHELON FORM OF A MATRIX

// Function for finding the row echelon form of a square matrix
//int row_ech_dense(denseMatrix* A) {}


//TODO: FINISH DETERMINANT COMPUTATION

// Function for computing the determinant of a square matrix
// using Gaussian elimination
//int det_dense(denseMatrix* A, double* ret) {}



// ADVANCED MATH OPERATIONS


// Function for inverting a matrix using Gauss-Jordan method
// Modified from: https://rosettacode.org/wiki/Gauss-Jordan_matrix_inversion#C
// Returns 0 if operation is successful 1 otherwise
int inv_dense(denseMatrix* A, denseMatrix* ret) {
	// Check that the matrix dimensions match
	if (!(A->n == ret->n && A->m == ret->m)) {
		printf("\nERROR CODE 2: Inversion failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that the matrix is symmetric
	if (!(A->n == A->m)) {
		printf("\nERROR: Inversion failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || ret->proper_init) {
		printf("\nERROR: Inversion failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Create an identity matrix of correct size which will converted to the inverse
	denseMatrix* I = eye_dense(A->n, A->n);
	// Check that the operation was successful
	if (I->proper_init) {
		printf("\nERROR: Inversion failed as generating an identity matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
									
		// Even in case of error free the allocated memory
		free_denseMatrix(I);
		
		return 1;
	}
	
	// As we don't want to destroy A create a copy of it
	denseMatrix* _A = alloc_denseMatrix(A->n, A->n);
	if (_A->proper_init) {
		printf("\nERROR: Inversion failed as allocating memory for a copy of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
											
		// Even in case of error free the allocated memory
		free_denseMatrix(I);
		free_denseMatrix(_A);
		
		return 1;
	}
	
	// Copy the contents of A into _A
	if (copy_dense(_A, A)) {
		printf("\nERROR: Inversion failed as copying of the contents of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
													
		// Even in case of error free the allocated memory
		free_denseMatrix(I);
		free_denseMatrix(_A);
		
		return 1;
	}
	
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
			printf("\nERROR: Given matrix is singular and thus cannot have an inverse\n");
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
	if (copy_dense(ret, I)) {
		printf("\nERROR: Inversion failed as copying the contents to ret failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
													
		// Even in case of error free the allocated memory
		free_denseMatrix(I);
		free_denseMatrix(_A);
		
		return 1;
	}
	
	// Free the allocated temporary memory
	free_denseMatrix(I);
	free_denseMatrix(_A);
	
	return 0;
}


// Function for Cholensky decomposition A = LL^T. Works only for s.p.d matrices
// Returns 0 if operation is successful 1 otherwise
int chol_dense(denseMatrix* A, denseMatrix* L) {
	// Check that the matrix dimensions match
	if (!(A->n == L->n && A->m == L->m)) {
		printf("\nERROR: Cholensky failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is symmetric
	if (!(A->n == A->m)) {
		printf("\nERROR: Cholensky failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || L->proper_init) {
		printf("\nERROR: Cholensky failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy A create a copy of it
	denseMatrix* _A = alloc_denseMatrix(A->n, A->n);
	if (_A->proper_init) {
		printf("\nERROR: Cholensky failed as allocating memory for a copy of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
	
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		
		return 1;
	}
	
	// Copy the contents of A into _A
	if (copy_dense(_A, A)) {
		printf("\nERROR: Cholensky failed as copying of the contents of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		
		return 1;
	}
	
	// Go over the rows of A
	for (int i = 0; i < _A->n - 1; i++) {
		// Allocate memory for update values for L
		denseMatrix* a21_T = alloc_denseMatrix(1, _A->n - (i + 1));
		denseMatrix* A22_tmp = alloc_denseMatrix(_A->n - (i + 1), _A->n - (i + 1));
		
		// Get the needed subarrays
		double a11;
		int a_success = _apply_dense(_A, &a11, i, i);
		if (a11 <= 0) {
			printf("\nERROR: Cholensky failed as the given matrix is not symmetric positive definite\n");
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
		if (A22->proper_init || a21->proper_init || A22_tmp->proper_init || a21_T->proper_init || a_success) {
			printf("\nERROR: Cholensky failed as some subarray allocations failed\n");
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
		int T_success = transpose_dense(a21, a21_T);
		int m_success = mult_dense(a21, a21_T, A22_tmp);
		int s1_success = smult_dense(A22_tmp, A22_tmp, 1. / a11);
		int d_success = diff_dense(A22, A22_tmp, A22);
		
		double l11 = sqrt(a11);
		int s2_success = smult_dense(a21, a21, 1. / l11);
		// Check that the operations were successful
		if (T_success || m_success || s1_success || d_success || s2_success) {
			printf("\nERROR: Cholensky failed as there was a problem with some math operation\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
										
			// Even in case of error free the allocated memory
			free_denseMatrix(_A);
			free_denseMatrix(a21_T);
			free_denseMatrix(A22_tmp);
			free_denseMatrix(a21);
			free_denseMatrix(A22);
		
			return 1;
		}
		
		// Place the values back to the arrays
		int p1_success = _place_dense(L, l11, i, i);
		int p2_success = _place_subarray_dense(L, a21, i + 1, _A->n, i, i + 1);
		int p3_success = _place_subarray_dense(_A, A22, i + 1, _A->n, i + 1, _A->n);
		// Check that the placing was successful
		if (p1_success || p2_success || p3_success) {
			printf("\nERROR: Cholensky failed as there was a problem with placing some subarray\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
										
			// Even in case of error free the allocated memory
			free_denseMatrix(_A);
			free_denseMatrix(a21_T);
			free_denseMatrix(A22_tmp);
			free_denseMatrix(a21);
			free_denseMatrix(A22);
		
			return 1;
		}
		
		// Free the temporary arrays;
		free_denseMatrix(a21);
		free_denseMatrix(a21_T);
		free_denseMatrix(A22);
		free_denseMatrix(A22_tmp);
	}
	
	// Place the last value into the array
	double a11;
	int a_success = _apply_dense(_A, &a11, L->n - 1, L->n - 1);
	if (a11 <= 0) {
		printf("\nERROR: Cholensky failed as the given matrix is not symmetric positive definite\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		
		return 1;
	}
	
	if (a_success || _place_dense(L, sqrt(a11), L->n - 1, L->n - 1)) {
		printf("\nERROR: Cholensky failed as there was a problem with placing some subarray\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		
		return 1;
	}
	
	// Free the copy of A
	free_denseMatrix(_A);
	
	return 0;
}


// Function for generic PLU decomposition. Works for invertible (nonsingular) matrices
// NOTE! The passed arguments P, L and U should all be initialized to 0
// Returns 0 if operation is successful 1 otherwise
int PLU_dense(denseMatrix* A, denseMatrix* P, denseMatrix* L, denseMatrix* U) {
	// Check that the matrix dimensions match
	if (!(A->n == L->n && A->m == L->m && A->n == U->n && A->m == U->m && A->n == P->n && A->m == P->m)) {
		printf("\nERROR: PLU failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is symmetric
	if (!(A->n == A->m)) {
		printf("\nERROR: PLU failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are large enough
	if (!(A->n > 1)) {
		printf("\nERROR: PLU failed as the matrix isn't large enough\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || L->proper_init) {
		printf("\nERROR: PLU failed as some matrix isn't properly allocated\n");
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
		printf("\nERROR: PLU failed as allocating memory for a copy of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		free_denseMatrix(P2);
		free_denseMatrix(a21);
		
		return 1;
	}
	
	// Copy the contents of A into _A
	if (copy_dense(_A, A)) {
		printf("\nERROR: PLU failed as copying of the contents of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		free_denseMatrix(P2);
		free_denseMatrix(a21);
		
		return 1;
	}
	
	// Turn the permutation matrix P into an identity matrix
	if (init_eye_dense(P)) {
		printf("\nERROR: PLU failed as initializing P as an identity matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		free_denseMatrix(P2);
		free_denseMatrix(a21);
		
		return 1;
	}
	
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
			printf("\nERROR: Given matrix is singular and thus cannot have a PLU decomposition\n");
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
			printf("\nERROR: PLU decomposition failed as there was a problem with temporary array allocation\n");
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
		int T = transpose_dense(P2, P2_T);
		int m1 = mult_dense(P2_T, _A, PA);
		int a = _apply_dense(PA, &a11_2, k, k);
		denseMatrix* a21_2 = _subarray_dense(PA, k + 1, _A->n, k, k + 1);
		denseMatrix* a21_tmp = _subarray_dense(a21, k, _A->n, 0, 1);
		denseMatrix* a12 = _subarray_dense(PA, k, k + 1, k + 1, _A->n);
		denseMatrix* A22 = _subarray_dense(PA, k + 1, _A->n, k + 1, _A->n);
		denseMatrix* P22 = _subarray_dense(P, k, _A->n, k, _A->n);
		denseMatrix* P2_tmp = _subarray_dense(P2, k, _A->n, k, _A->n);
		denseMatrix* P2_tmp_T = _subarray_dense(P2_T, k, _A->n, k, _A->n);
		// Check that the operations were successful
		if (a21_2->proper_init || a21_tmp->proper_init || a12->proper_init || A22->proper_init || P22->proper_init || P2_tmp->proper_init || P2_tmp_T->proper_init || T || m1 || a) {
			printf("\nERROR: PLU decomposition failed as some subarray allocation failed\n");
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
		int p1 = _place_dense(L, (double)1.0, k, k);
		int p2 = _place_dense(U, a11_2, k, k);
		int p3 = _place_subarray_dense(U, a12, k, k + 1, k + 1, _A->n);
		
		// Update _A
		int m2 = mult_dense(a21_2, a12, A22_tmp);
		int s1 = smult_dense(A22_tmp, A22_tmp, 1.0 / a11_2);
		int d = diff_dense(A22, A22_tmp, A22);
		int p4 = _place_subarray_dense(_A, A22, k + 1, _A->n, k + 1, _A->n);
		
		// Update P and L
		int m3 = 0, p5 = 0, m4 = 0, s2 = 0, p6 = 0;
		if (k > 0) {
			m3 = mult_dense(P22, P2_tmp, P22);
			p5 = _place_subarray_dense(P, P22, k, _A->n, k, _A->n);
			m4 = mult_dense(P2_tmp_T, a21_tmp, a21_tmp);
			s2 = smult_dense(a21_tmp, a21_tmp, 1.0 / a11);
			p6 = _place_subarray_dense(L, a21_tmp, k, _A->n, k - 1, k); 
		} 
		
		// Update a21 and a11
		int p7 = _place_subarray_dense(a21, a21_2, k + 1, _A->n, 0, 1);
		a11 = a11_2;
		
		// Check that everything was successful
		if (p1 || p2 || p3 || p4 || p5 || p6 || p7 || m2 || m3 || m4 || s1 || s2 || d) {
			printf("\nERROR: PLU decomposition failed as an error occured in some math operation\n");
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
	int a1_success = _apply_dense(_A, &A_nn, _A->n - 1, _A->n - 1);
	int p1_success = _place_dense(U, A_nn, _A->n - 1, _A->n - 1);
	// For L
	int p2_success = _place_dense(L, (double)1.0, _A->n - 1, _A->n - 1);
	double P2_nn;
	int a2_success = _apply_dense(P2, &P2_nn, _A->n - 1, _A->n - 1);
	double a21_n;
	int a3_success = _apply_dense(a21, &a21_n, _A->n - 1, 0);
	int p3_success = _place_dense(L, P2_nn * a21_n / a11, _A->n - 1, _A->n - 2);
	
	// Check that the operations were successful
	if (a1_success || p1_success || p2_success || a2_success || a3_success || p3_success) {
		printf("\nERROR: PLU decomposition failed as an error occured in some math operation\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(_A);
		free_denseMatrix(P2);
		free_denseMatrix(a21);
		
		return 1;
	}
	
	
	// Free rest of the arrays
	free_denseMatrix(_A);
	free_denseMatrix(P2);
	free_denseMatrix(a21);
	
	return 0;
}


// Function for solving a system of linear equations Lx = b, where 
// L is an invertible lower triangular matrix
// Returns 0 if operation is successful 1 otherwise
int trilsolve_dense(denseMatrix* L, denseMatrix* x, denseMatrix* b) {
	// Check that the matrix dimensions match
	if (!(L->m == x->n && L->n == b->n && x->m == b->m)) {
		printf("\nERROR: Trilsolve failed as matrix dimensions don't match\n");
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
		printf("\nERROR: Trilsolve failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (L->proper_init || x->proper_init || b->proper_init) {
		printf("\nERROR: Trilsolve failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy L or b create copies of them
	denseMatrix* _L = alloc_denseMatrix(L->n, L->n);
	denseMatrix* _b = alloc_denseMatrix(b->n, 1);
	
	// Check that the allocation was successful
	if (_L->proper_init || _b->proper_init) {
		printf("\nERROR: Trilsolve failed as allocating memory for a copy of L failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
		// Even in case of error free the allocated memory
		free_denseMatrix(_L);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// Copy the contents of L into _L
	int c1_success = copy_dense(_L, L);
	int c2_success = copy_dense(_b, b);
	if (c1_success || c2_success) {
		printf("\nERROR: Trilsolve failed as copying of the contents of L failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_L);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// The main loop body
	for (int i = 0; i < _L->n - 1; i++) {
		// Split the linear system into blocks and allocate memory
		// for needed subarrays
		
		// For L
		double l11;
		int a1_success = _apply_dense(_L, &l11, i, i);
		denseMatrix* l21 = _subarray_dense(_L, i + 1, _L->n, i, i + 1);
		// Check that l11 is not 0
		if (!(l11 != 0)) {
			printf("\nERROR: Trilsolve failed as L is not invertible\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_L);
			free_denseMatrix(_b);
			free_denseMatrix(l21);
			
			return 1;
		}
		
		// For b
		double b1;
		int a2_success = _apply_dense(_b, &b1, i, 0);
		denseMatrix* b2 = _subarray_dense(_b, i + 1, _b->n, 0, 1);
		
		// Check that the operations were successful
		if (l21->proper_init || b2->proper_init || a1_success || a2_success) {
			printf("\nERROR: Trilsolve failed as there was an error in subarray allocation\n");
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
		int p_success = _place_dense(x, x_i, i, 0);
		
		// Update b
		int s_success = smult_dense(l21, l21, x_i);
		int d_success = diff_dense(b2, l21, b2);
		int ps_success = _place_subarray_dense(_b, b2, i + 1, _b->n, 0, 1);
		
		// Check that the operations were successful
		if (p_success || s_success || d_success || ps_success) {
			printf("\nERROR: Trilsolve failed as there was an error with some math operation\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
						
			// Even in case of error free the allocated memory
			free_denseMatrix(_L);
			free_denseMatrix(_b);
			free_denseMatrix(l21);
			free_denseMatrix(b2);
				
			return 1;
		}
		// Free temporary arrays
		free_denseMatrix(l21);
		free_denseMatrix(b2);
	}
	
	// Update the final element of x
	double l11;
	int a1 = _apply_dense(_L, &l11, _L->n - 1, _L->n - 1);
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
	int a2 = _apply_dense(_b, &b1, _b->n - 1, 0);
	int p = _place_dense(x, b1 / l11, x->n - 1, 0);
	if (a1 || a2 || p) {
		printf("\nERROR: Trilsolve failed as there was an error in subarray allocation\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
		// Even in case of error free the allocated memory
		free_denseMatrix(_L);
		free_denseMatrix(_b);
			
		return 1;
	}
	
	// Free allocated memory
	free_denseMatrix(_L);
	free_denseMatrix(_b);
	
	return 0;
}


// Function for solving a system of linear equations Ux = b, where 
// U is an invertible upper triangular matrix
// Returns 0 if operation is successful 1 otherwise
int triusolve_dense(denseMatrix* U, denseMatrix* x, denseMatrix* b) {
	// Check that the matrix dimensions match
	if (!(U->m == x->n && U->n == b->n && x->m == b->m)) {
		printf("\nERROR: Triusolve failed as matrix dimensions don't match\n");
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
		printf("\nERROR: Triusolve failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (U->proper_init || x->proper_init || b->proper_init) {
		printf("\nERROR: Triusolve failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy L or b create copies of them
	denseMatrix* _U = alloc_denseMatrix(U->n, U->n);
	denseMatrix* _b = alloc_denseMatrix(b->n, 1);
	
	// Check that the allocation was successful
	if (_U->proper_init || _b->proper_init) {
		printf("\nERROR: Triusolve failed as allocating memory for a copy of L failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			
		// Even in case of error free the allocated memory
		free_denseMatrix(_U);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// Copy the contents of U into _U and b into _b
	int c1_success = copy_dense(_U, U);
	int c2_success = copy_dense(_b, b);
	if (c1_success || c2_success) {
		printf("\nERROR: Trilsolve failed as copying of the contents of L failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_U);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// The main loop body
	for (int i = _U->n - 1; i > 0; i--) {
		// Split the linear system into blocks and allocate memory
		// for needed subarrays
		
		// For U
		double u22;
		int a1_success = _apply_dense(_U, &u22, i, i);
		// Check that u22 is not 0
		if (!(u22 != 0)) {
			printf("\nERROR: Triusolve failed as U is not invertible\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_U);
			free_denseMatrix(_b);
			
			return 1;
		}	

		denseMatrix* u12 = _subarray_dense(_U, 0, i, i, i + 1);
		
		// For b
		double b2;
		int a2_success = _apply_dense(_b, &b2, i, 0);
		denseMatrix* b1 = _subarray_dense(_b, 0, i, 0, 1);
		
		// Check that the operations were successful
		if (u12->proper_init || b1->proper_init || a1_success || a2_success) {
			printf("\nERROR: Triusolve failed as there was an error in subarray allocation\n");
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
		int p_success = _place_dense(x, x_i, i, 0);
		
		// Update b
		int s_success = smult_dense(u12, u12, x_i);
		int d_success = diff_dense(b1, u12, b1);
		int ps_success = _place_subarray_dense(_b, b1, 0, i, 0, 1);
		
		// Check that the operations were successful
		if (p_success || s_success || d_success || ps_success) {
			printf("\nERROR: Triusolve failed as there was an error with some math operation\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
			// Even in case of error free the allocated memory
			free_denseMatrix(_U);
			free_denseMatrix(_b);
			free_denseMatrix(u12);
			free_denseMatrix(b1);
			
			return 1;
		}
		
		// Free temporary arrays
		free_denseMatrix(u12);
		free_denseMatrix(b1);
	}

	// Handle the final element of x
	double u22;
	int a1 = _apply_dense(_U, &u22, 0, 0);
	// Check that u22 is not 0
	if (!(u22 != 0)) {
		printf("\nERROR: Triusolve failed as U is not invertible\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				
		// Even in case of error free the allocated memory
		free_denseMatrix(_U);
		free_denseMatrix(_b);
			
		return 1;
	}
	double b2;
	int a2 = _apply_dense(_b, &b2, 0, 0);
	int p = _place_dense(x, b2 / u22, 0, 0);
	// Check that the operations were successful
	if (a1 || a2 || p) {
		printf("\nERROR: Triusolve failed as there was an error in subarray allocation\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
					
		// Even in case of error free the allocated memory
		free_denseMatrix(_U);
		free_denseMatrix(_b);
			
		return 1;
	}

	// Free allocated memory
	free_denseMatrix(_U);
	free_denseMatrix(_b);
	
	return 0;
}


// Function for solving a system of linear equations of form Ax = b
// using PLU decomposition of the matrix A
// Returns 0 if operation is successful 1 otherwise
int linsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b) {
	// Check that the matrix dimensions match
	if (!(A->m == x->n && A->n == b->n && x->m == b->m)) {
		printf("\nERROR: Linsolve failed as matrix dimensions don't match\n");
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
	if (!(A->n == A->m)) {
		printf("\nERROR: Linsolve failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || x->proper_init || b->proper_init) {
		printf("\nERROR: Linsolve failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Allocate memory for the P, L and U matrices passed as argument to
	// the plu function (and temporary array y)
	denseMatrix* P = alloc_denseMatrix(A->n, A->n);
	denseMatrix* P_T = alloc_denseMatrix(A->n, A->n);
	denseMatrix* L = alloc_denseMatrix(A->n, A->n);
	denseMatrix* U = alloc_denseMatrix(A->n, A->n);
	denseMatrix* y = alloc_denseMatrix(A->n, 1);
	
	// Check that the allocations were successful
	if (P->proper_init || L->proper_init || U->proper_init || y->proper_init) {
		printf("\nERROR: Linsolve failed as there was an error in temporary array allocation\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(P);
		free_denseMatrix(P_T);
		free_denseMatrix(L);
		free_denseMatrix(U);
		free_denseMatrix(y);
		
		return 1;
	}
	
	// As we don't want to destroy b create a copy of it 
	denseMatrix* _b = alloc_denseMatrix(A->n, 1);
	int c_success = copy_dense(_b, b);
	if (_b->proper_init || c_success) {
		printf("\nERROR: Linsolve failed as there was an error in copying the contents of b\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(P);
		free_denseMatrix(P_T);
		free_denseMatrix(L);
		free_denseMatrix(U);
		free_denseMatrix(y);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// Call the PLU function to find a decomposition of form A = PLU
	// Then the system of equations becomes LUx = P^(T)b (permutation matrix is unitary)
	// This can be solved in two steps: Ly = P^(T)b => Ux = y
	// Final solution should be x = Px
	
	if (PLU_dense(A, P, L, U)) {
		printf("\nERROR: Linsolve failed as there was an error with the PLU decomposition\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(P);
		free_denseMatrix(P_T);
		free_denseMatrix(L);
		free_denseMatrix(U);
		free_denseMatrix(y);
		free_denseMatrix(_b);
		
		return 1;
	} 
	
	// Solve for y
	int T_success = transpose_dense(P, P_T);
	int m_success = mult_dense(P_T, _b, _b);
	int tril = trilsolve_dense(L, y, _b);
	if (T_success || m_success || tril) {
		printf("\nERROR: Linsolve failed as an error arose when solving Ly = P^(T)b\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(P);
		free_denseMatrix(P_T);
		free_denseMatrix(L);
		free_denseMatrix(U);
		free_denseMatrix(y);
		free_denseMatrix(_b);
		
		return 1;
	} 
	
	// Solve for x
	if (triusolve_dense(U, x, y)) {
		printf("\nERROR: Linsolve failed as an error arose when solving Ux = y\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(P);
		free_denseMatrix(P_T);
		free_denseMatrix(L);
		free_denseMatrix(U);
		free_denseMatrix(y);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// Permute x 
	if (mult_dense(P, x, x)) {
		printf("\nERROR: Linsolve failed as an error arose permuting x\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Even in case of error free the allocated memory
		free_denseMatrix(P);
		free_denseMatrix(P_T);
		free_denseMatrix(L);
		free_denseMatrix(U);
		free_denseMatrix(y);
		free_denseMatrix(_b);
		
		return 1;
	}
	
	// Free allocated memory
	free_denseMatrix(P);
	free_denseMatrix(P_T);
	free_denseMatrix(L);
	free_denseMatrix(U);
	free_denseMatrix(y);
	free_denseMatrix(_b);
	
	return 0;
}


// TODO: FINISH CONJUGATE GRADIENT METHOD FOR SOLVING Ax = b

// Function for solving an approximate solutions to the linear system Ax = b
// by minimizing an equivalent quadratic (convex) problem J(u) = 1/2 u^T Au - b^T u
// using conjugate gradient method with A-orthogonal search directions.
// Requires that A is symmetrix positive definite
// Returns 0 if operation is successful 1 otherwise
//int cgsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b) {}


// TODO: FINISH CHOLENSKY DECOMPOSITION BASED METHOD FOR SOLVING Ax = b

// Function for solving a system of equations Ax = b using Cholensky 
// decomposition of a symmetric positive definite matrix A.
// Returns 0 if operation is successful 1 otherwise
//int cholsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b) {}


// TODO: FINISH MATRIX INVERSE BASED METHOD FOR SOLVING Ax = b

// Function for solving a system of equations Ax = b by computing x = A^(-1) b
// Requires that A is invertible. Mainly for benchmarking purposes
// Returns 0 if operation is successful 1 otherwise
//int invsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b) {}


// TODO: FINISH MATRIX EXPONENTIATION


// TODO: FINISH EIGENDECOMPOSITION

// Function for computing the eigendecomposition (that is A = SES^-1 
// where S has the eigenvectors of A as columns and E has the eigenvalues
// of A on the diagonal) of a given matrix A
// Computed using the QR-algorithm
// Returns 0 if operation is successful 1 otherwise
//int eig_dense(denseMatrix* A, denseMatrix* S, denseMatrix* E, denseMatrix* S_inv) {}


// TODO: FINISH SCHUR FACTORIZATION

// Function for computing the Schur factorization of A i.e. A = QUQ^T
// where Q is unitary and U is upper triangular. This is useful as 
// A and U must be similar meaning that the eigenvalues of A must appear
// on the diagonal of U. Also unlike with eigendecomposition Schur factorization
// can be found for every square matrix A.



// TESTING FUNCTIONS

// Function for timing the solution to system of equations Ax = b
// Cuts the program execution if the system is not solvable with given 
// solution method. Only frees the passed argument matrices so there shouldn't
// be any others allocated
double solve_timer_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b, 
					     int (*solve)(denseMatrix*, denseMatrix*, denseMatrix*)) {
	clock_t begin = clock();
	if ((*solve)(A, x, b)) {
		printf("\nERROR: Couldn't solve the system\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		free_denseMatrix(A);
		free_denseMatrix(x);
		free_denseMatrix(b);
		
		exit(0);
	}
	clock_t end = clock();
	
	return (double)(end - begin) / CLOCKS_PER_SEC;	   
}


// Main function for testing the library
// To compile this: 
// - navigate to C-Libraries folder
// - compile: gcc -fopenmp -Wall "Linear Algebra/dense_matrix.c" general.c -lm -o matrix.o
// - run: ./matrix.o
// - valgrind: valgrind --leak-check=full --undef-value-errors=no -v ./matrix.o
int main() {
	// Define a double arrays
	double arr1[9] = {1.0, 2.0, 3.0,
					  4.0, 5.0, 6.0,
					  7.0, 8.0, 9.0};
	double arr2[9] = {1.0, 1.0, 1.0,
					  2.0, 2.0, 2.0,
					  3.0, 3.0, 3.0};
					  
	// Convert to denseMatrix
	denseMatrix* A = conv_to_denseMatrix(arr1, 3, 3);
	denseMatrix* B = conv_to_denseMatrix(arr2, 3, 3);
	denseMatrix* C = alloc_denseMatrix(3, 3);
	
	// Do elementwise multiplication of A.*B and store in A
	hprod_dense(A, B, A);
	// Print the result (should be A = [1 2 3; 8 10 12; 21 24 27])
	printf("\nResult of multiplication A.*B\n");
	print_dense(A);

	// Do elementwise division A./B and store in A
	hdiv_dense(A, B, A);
	// Print the result (should return A back to initial)
	printf("\nResult of division A./B\n");
	print_dense(A);
	
	// Do normal matrix multiplication A*B and store in C
	mult_dense(A, B, C);
	// Print the result (should be C = [14 14 14; 32 32 32; 50 50 50]
	printf("\nResult of matrix product A*B\n");
	print_dense(C);
	
	// Compute the elementwise power for A and store it in A
	hpow_dense(A, A, 2);
	// Print the result (should be A = [1 4 9; 16 25 36; 49 64 81]
	printf("\nResult of matrix power A.^2\n");
	print_dense(A);
	
	// Compute the power of B and store it in C
	pow_dense(B, C, 2);
	// Print the result (should be A = [6 6 6; 12 12 12; 18 18 18]
	printf("\nResult of matrix power B^2\n");
	print_dense(C);
	
	// Try to compute the inverse of C and store it in C (should raise an error)
	inv_dense(C, C);
	
	// Define an invertible matrix and invert it. Store in E
	double arr5[9] = {9.0, 4.0, 7.0,
					  4.0, 5.0, 4.0,
					  7.0, 4.0, 9.0};
	denseMatrix* D = conv_to_denseMatrix(arr5, 3, 3);
	denseMatrix* E = alloc_denseMatrix(3, 3);
	inv_dense(D, E);
	//Print the result (should be E = [0.3 -0.083 -0.2; -0.083 0.33 -0.083; -0.2 -0.083 0.3]
	printf("\nResult of matrix inverse D^(-1)\n");
	print_dense(E);
	
	// Compute the cholensky decomposition of C (should raise an error)
	chol_dense(C, C);
	
	// Compute the cholensky decomposition of D and store in F
	denseMatrix* F = alloc_denseMatrix(3, 3);
	chol_dense(D, F);
	// Print the results (should be F = [3 0 0; 1.333 1.795 0; 2.333 0.495 1.819])
	printf("\nThe lower triangular from Cholensky decomp D = LL^(T)\n");
	print_dense(F);
	
	// Allocate memory for PLU decomp
	denseMatrix* P = alloc_denseMatrix(3, 3);
	denseMatrix* L = alloc_denseMatrix(3, 3);
	denseMatrix* U = alloc_denseMatrix(3, 3);
	
	// Compute the PLU decomposition for D
	PLU_dense(D, P, L, U);
	// Print the results
	printf("\nThe permutation matrix from PLU of D\n");
	print_dense(P);  // Should be P = [1 0 0; 0 1 0; 0 0 1]
	printf("\nThe lower triag from PLU of D\n");
	print_dense(L);  // Should be L = [1 0 0; 0.444 1 0; 0.788 0.28 1]
	printf("\nThe upper triag from PLU of D\n");
	print_dense(U);  // Should be P = [9 4 7; 0 3.222 0.888; 0 0 3.3]
	
	// Solve system of equations Lx1 = b where b is
	double arr6[3] = {1.0, 2.0, 3.0};
	denseMatrix* b = conv_to_denseMatrix(arr6, 3, 1);
	denseMatrix* x1 = alloc_denseMatrix(3, 1);
	trilsolve_dense(L, x1, b);
	// Print the results (should be x1 = [1.0; 1.556; 1.792])
	printf("\nSolution to system of equations L*x1 = b\n");
	print_dense(x1);

	// Solve system of equations Ux2 = b
	denseMatrix* x2 = alloc_denseMatrix(3, 1);
	triusolve_dense(U, x2, b);	
	// Print the results (should be x2 = ...)
	printf("\nSolution to system of equations U*x2 = b\n");
	print_dense(x2);

	// Test by multiplication
	mult_dense(U, x2, x2);
	// Print the results (should be x2 = b)
	printf("\nResult of multiplication U*x2\n");
	print_dense(x2);

	// Solve system Dx3 = b
	denseMatrix* x3 = alloc_denseMatrix(3, 1);
	linsolve_dense(D, x3, b);
	// Print the results (should be x2 = ...)
	printf("\nSolution to system of equations D*x3 = b\n");
	print_dense(x3);

	// Test by multiplication
	mult_dense(D, x3, x3);
	// Print the results (should be x2 = b)
	printf("\nResult of multiplication D*x3\n");
	print_dense(x3);
	
	// Free the allocated matrices
	free_denseMatrix(A);
	free_denseMatrix(B);
	free_denseMatrix(C);
	free_denseMatrix(D);
	free_denseMatrix(E);
	free_denseMatrix(F);
	free_denseMatrix(P);
	free_denseMatrix(L);
	free_denseMatrix(U);
	free_denseMatrix(b);
	free_denseMatrix(x1);
	free_denseMatrix(x2);
	free_denseMatrix(x3);
}

