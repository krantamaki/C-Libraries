#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "../general.h"
#include "dense_matrix.h"


// (UNDER CONSTRUCTION!)
// This is a general linear algebra library using dense matrices 
// with double precision floating point numbers.
// Optimized to allow multithreading, vectorization (256 SIMD) and ILP, 
// but lacks prefetching, improved cache management and proper register 
// reuse


// GENERAL FUNCTIONS

// Function for allocating memory for wanted sized denseMatrix
denseMatrix alloc_denseMatrix(int n, int m) {
	// Check that given dimensions are positive
	if (!(n > 0 && m > 0)) {
		printf("\nERROR: Matrix dimesions must be positive\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix ret;
		ret.n = n;
		ret.m = m;
		ret.vects_per_row = _ceil(m, DOUBLE_ELEMS);
		ret.data = NULL;
		ret.proper_init = 1;
		return ret;
	}
	
	// Number of vectors per row
	int vect_num = _ceil(m, DOUBLE_ELEMS);
	// Total number of vectors
	size_t len = n * vect_num;
	
	// Allocate aligned memory
	void* tmp = 0;
	if (posix_memalign(&tmp, sizeof(double4_t), len * sizeof(double4_t))) {
		printf("\nERROR: Aligned memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix ret;
		ret.n = n;
		ret.m = m;
		ret.vects_per_row = vect_num;
		ret.data = NULL;
		ret.proper_init = 1;
		return ret;
	}
	
	// Fill the matrix
	denseMatrix ret;
	ret.n = n;
	ret.m = m;
	ret.vects_per_row = vect_num;
	ret.data = (double4_t*)tmp;
	ret.proper_init = 0;
	
	return ret;
}

// Function for converting a double array to denseMatrix
// Takes the first n * m elements from double array so it is assumed
// that len(arr) >= n * m
denseMatrix conv_to_denseMatrix(double* arr, int n, int m) {
	// Check that given dimensions are positive
	if (!(n > 0 && m > 0)) {
		printf("\nERROR: Matrix dimesions must be positive\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix ret;
		ret.n = n;
		ret.m = m;
		ret.vects_per_row = _ceil(m, DOUBLE_ELEMS);
		ret.data = NULL;
		ret.proper_init = 1;
		return ret;
	}
	
	// Allocate memory for the denseMatrix
	denseMatrix ret = alloc_denseMatrix(n, m);
	
	// Check that allocation was succesful
	if (ret.proper_init) {
		printf("\nERROR: Cannot convert to denseMatrix as memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	const int vect_num = ret.vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			// Go over the elements in each vector
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				int j = vect * DOUBLE_ELEMS + elem;
				ret.data[vect_num * i + vect][elem] = i < n && j < m ? arr[i * m + j] : 0.0;
			}
		}
	}
	
	return ret;
}

// Function for getting an individual value from a denseMatrix
// Assumes 0 <= i < n and 0 <= j < m and that matrix is properly allocated
double _apply_dense(denseMatrix A, int i, int j) {
	// Find the proper vector and element in said vector for column j
	int vect = j / DOUBLE_ELEMS;  // Integer division defaults to floor
	int elem = j % DOUBLE_ELEMS;
	
	return A.data[A.vects_per_row * i + vect][elem];
}

// Function for placing an individual value into a wanted place in a denseMatrix
// Returns 0 if operation is successful, 1 otherwise
int _place_dense(denseMatrix A, int i, int j double val) {
	// Check that the wanted index is viable for the matrix
	if (!(i < A.n && j < A.m)) {
		printf("\nERROR: Given indeces exceed the dimensions of the matrix\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Find the proper vector and element in said vector for column j
	int vect = j / DOUBLE_ELEMS;  // Integer division defaults to floor
	int elem = j % DOUBLE_ELEMS;
	A.data[vect_num * i + vect][elem] = val;
	
	return 0;
}

// Function for getting an subarray of an existing denseMatrix
denseMatrix _subarray_dense(denseMatrix A, int n_start, int n_end, int m_start int m_end) {
	// Check that the dimensions are proper
	if (!(n_start < n_end && n_end <= A.n && m_start < m_end && m_end <= A.m)) {
		printf("\nERROR: The start index of a slice must be smaller than end index and end index must be equal or smaller than matrix dimension");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		denseMatrix ret;
		ret.n = n_end - n_start;
		ret.m = m_end - m_start;
		ret.vects_per_row = _ceil(m_end - m_start, DOUBLE_ELEMS);
		ret.data = NULL;
		ret.proper_init = 1;
		return ret;
	}
	
	denseMatrix ret = alloc_denseMatrix(n_end - n_start, m_end - m_start);
	
	// Check that allocation was successful
	if (ret.proper_init) {
		printf("\nERROR: Subarray cannot be retrieved as memory allocation failed\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	const int vect_num = ret.vects_per_row;
	// Find the end and start vectors of wanted columns
	const int vect_start = m_start / DOUBLE_ELEMS;  // Integer division defaults to floor
	const int vect_end = _ceil(m_end, DOUBLE_ELEMS);
	
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i0 = 0; i0 < n_end - n_start; i0++) {
		int i = i0 + n_start;
		int vect = vect_start;
		int elem = m_start % DOUBLE_ELEMS;
		// Go over the vectors for each row
		for (int vect0 = 0; vect0 < vect_num; vect0++) {
			// Go over the elements in each vector
			for (int elem0 = 0; elem0 < DOUBLE_ELEMS; elem0++) {
				if (elem == DOUBLE_ELEMS) {
					vect++;
					elem = 0;
				}
				int j = vect * DOUBLE_ELEMS + elem;
				ret.data[vect_num * i0 + vect0][elem0] = _apply_dense(A, i, j);
				elem++;
			} 
		}
	}
	
	return ret;
}


// BASIC MATH OPERATIONS

// Function for summing two matrices
// Returns 0 if operation is successful, 1 otherwise
int sum_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n == B.n && A.n = ret.n && A.m == B.m && A.m = ret.m)) {
		printf("\nERROR: Summation failed as matrix dimensions don't match\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || B.proper_init || ret.proper_init) {
		printf("\nERROR: Summation failed as some matrix isn't properly allocated\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int n = A.n;
	const int m = A.m;
	const int vect_num = A.vects_per_row;
	
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			ret.data[vect_num * i + vect] = A.data[vect_num * i + vect] + B.data[vect_num * i + vect];
		}
	}
	
	return 0;
}

// Function for negating a matrix
// Returns 0 if operation is successful, 1 otherwise
int negate_dense(denseMatrix A, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n = ret.n && A.m = ret.m)) {
		printf("\nERROR: Negation failed as matrix dimensions don't match\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || ret.proper_init) {
		printf("\nERROR: Negation failed as some matrix isn't properly allocated\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const double4_t neg = {(double)-1.0, (double)-1.0, (double)-1.0, (double)-1.0}; 
	
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			ret.data[vect_num * i + vect] = neg * A.data[vect_num * i + vect];
		}
	}
	
	return 0;
}

// Function for taking the difference between two matrices (i.e. A - B))
// Returns 0 if operation is successful, 1 otherwise
int diff_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n == B.n && A.n = ret.n && A.m == B.m && A.m = ret.m)) {
		printf("\nERROR: Difference failed as matrix dimensions don't match\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || B.proper_init || ret.proper_init) {
		printf("\nERROR: Difference failed as some matrix isn't properly allocated\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Negate B so it can be summed with A
	if (negate_dense(B, B)) {
		printf("\nERROR: Negation step of difference failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Sum with A
	if (sum_dense(A, B, ret)) {
		printf("\nERROR: Summation step of difference failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	return 0;
}

// Function for transposing a given matrix
// Returns 0 if operation is successful 1 otherwise
int transpose_dense(denseMatrix A, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n = ret.m && A.m = ret.n)) {
		printf("\nERROR: Transpose failed as matrix dimensions don't match\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || ret.proper_init) {
		printf("\nERROR: Transpose failed as some matrix isn't properly allocated\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int vect_num0 = ret.vects_per_row;
	const int vect_num = A.vects_per_row;
	// Go over the rows of A
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A.n; i++) {
		// Go over the vectors for each row
		for (int vect = 0; vect < vect_num; vect++) {
			double4_t tmp = A.data[vect_num * i + vect];
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				int j = vect * DOUBLE_ELEMS + elem;
				int vect0 = i / DOUBLE_ELEMS;  // Integer division defaults to floor
				int elem0 = i % DOUBLE_ELEMS;
				
				ret[vect_num0 * j + vect0][elem0] = tmp[elem];
			}
		}
	}
	
	
	return 0;
}

// Function for multiplying two matrices (i.e AB)
// Returns 0 if operation is successful 1 otherwise
int mult_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.m == B.n && A.n = ret.n && B.m = ret.m)) {
		printf("\nERROR: Multiplication failed as matrix dimensions don't match\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || B.proper_init || ret.proper_init) {
		printf("\nERROR: Multiplication failed as some matrix isn't properly allocated\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// For linear reading we need the transpose of B
	// so allocate memory for this transose
	denseMatrix B_T = alloc_denseMatrix(b.m, b.n);
	// Check that allocation was successful
	if (B_T.proper_init) {
		printf("\nERROR: Memory allocation for the transpose of B failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// and transpose B
	if (transpose_dense(B, B_T)) {
		printf("\nERROR: Failed to transpose B\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int vect_num = A.vects_per_row;
	// Go over the rows of A
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A.n; i++) {
		// Go over the rows of B_T
		for (int j = 0; j < B_T.n) {
			// Multiply the values for the rows element-wise and sum them together
			
			// Go over the vectors
			double4_t sum = {(double)0.0, (double)0.0, (double)0.0, (double)0.0};
			for (int vect = 0; vect < vect_num; vect++) {
				sum += A.data[vect_num * i + vect] * B_T.data[vect_num * j + vect];
			}
			double val = 0.0;
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				val += sum[elem];
			}
			
			// Place the value at the correct location
			if (_place_dense(ret, i, j, val)) {
				printf("\nERROR: Failed to place the computed value in the correct location\n");
				printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
				return 1;
			}
			
		}
	}
	
	return 0;
}

// Function for inverting a matrix
// Returns 0 if operation is successful 1 otherwise
int inv_dense(denseMatrix A, denseMatrix ret) {
	
}


// ADVANCED MATH OPERATIONS

// Function for Cholensky decomposition. Works only for s.p.d matrices
// Returns 1 if operation is successful 0 otherwise
int chol_dense(denseMatrix A, denseMatrix L) {
	
}

// Function for PLU decomposition
// Returns 1 if operation is successful 0 otherwise
int plu_dense(denseMatrix A, denseMatrix P, denseMatrix L, denseMatrix U) {
	
}



