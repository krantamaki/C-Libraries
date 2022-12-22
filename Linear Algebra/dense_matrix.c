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
// Optimized to allow multithreading, vectorization (256 bit SIMD) and ILP, 
// but lacks prefetching, improved cache management and proper register reuse


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


// Function for freeing the memory allocated for a denseMatrix
void free_denseMatrix(denseMatrix A) {
	// Free the data array
	free(A.data);
	// Free the main struct
	free(A);
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
int _apply_dense(denseMatrix A, double* ret, int i, int j) {
	// Check that the indexes are within proper range
	if (!(i >= 0 && i < A.n && j >= 0 && j < A.m)) {
		printf("\nERROR: Given indeces exceed the dimensions of the matrix\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Find the proper vector and element in said vector for column j
	int vect = j / DOUBLE_ELEMS;  // Integer division defaults to floor
	int elem = j % DOUBLE_ELEMS;
	*ret = A.data[A.vects_per_row * i + vect][elem];
	
	return 0;
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
	// Check that the matrix is properly allocated
	if (A.proper_init) {
		printf("\nERROR: Couldn't place the value as the matrix isn't properly allocated\n");
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
	// Check that the matrix is properly allocated
	if (A.proper_init) {
		printf("\nERROR: Couldn't retrieve a subarray as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
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


// (Naive) Function for placing an denseMatrix B into a wanted position in another denseMatrix A
int _place_subarray_dense(denseMatrix A, denseMatrix B, n_start, n_end, m_start, m_end) {
	// Check that the matrix is properly allocated
	if (A.proper_init || B.proper_init) {
		printf("\nERROR: Couldn't place the subarray as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the dimensions are proper for A
	if (!(n_start < n_end && n_end <= A.n && m_start < m_end && m_end <= A.m)) {
		printf("\nERROR: The start index of a slice must be smaller than end index and end index must be equal or smaller than matrix dimension");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the dimensinos are proper for B
	if (!(n_end - n_start = B.n && m_end - m_start == B.m)) {
		printf("\nERROR: Size of the slice must correspond with the dimensions of the placed subarray");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Go over the row values
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i0 = 0; i0 < n_end - n_start; i0++) {
		int i = i0 + n_start;
		// Go over the column values
		for (int j0 = 0; j0 < m_end - m_start; j0++) {
			int j = j0 + m_start;
			double B_ij = _apply_dense(B, i0, j0);
			_place_dense(A, i, j, B_ij);
		}
	}
	
	return 0;
}


// Function for generating a wanted sized identity matrix
denseMatrix eye_dense(int n, int m) {
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
	
	denseMatrix ret = alloc_denseMatrix(n, m);
	
	// Check that allocation was successful
	if (ret.proper_init) {
		printf("\nERROR: Indentity matrix cannot be created as memory allocation failed\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	const int vect_num = ret.vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < n; i++) {
		// Go over each vector on the row
		for (int vect = 0; vect < vects_num; vect++) {
			// Go over each element in the vector
			for (int elem = 0; elem < DOUBLE_ELEMS; elem++) {
				int j = vect * DOUBLE_ELEMS + elem;
				double val = i == j && j < m ? (double)1.0 : (double)0.0;
				ret.data[vect_num * i + vect][elem] = val;
			}
		}
	}
	
	return ret;
}


// Function for copying the values of one matrix (src) into another (dst)
// NOTE! Could be changed to memcpy implementation (although that would 
// probably be less eficient)
int _copy_dense(denseMatrix dst, denseMatrix src) {
	// Check that the matrix dimensions match
	if (!(dst.n == src.n && dst.m == src.m)) {
		printf("\nERROR: Copying failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (dst.proper_init || src.proper_init) {
		printf("\nERROR: Copying failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int vect_num = A.vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A.n; i++) {
		// Go over each vector on the row
		for (int vect = 0; vect < vect_num; vect++) {
			dst.data[vect_num * i + vect] = src.data[vect_num * i + vect];
		}
	}
	
	return 0;
}



// BASIC MATH OPERATIONS

// Function for summing two matrices
// Returns 0 if operation is successful, 1 otherwise
int sum_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n == B.n && A.n == ret.n && A.m == B.m && A.m == ret.m)) {
		printf("\nERROR: Summation failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || B.proper_init || ret.proper_init) {
		printf("\nERROR: Summation failed as some matrix isn't properly allocated\n");
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


// Function for multiplying a denseMatrix with a scalar
// Returns 0 if operation is successful 1 otherwise
int smult_dense(denseMatrix A, denseMatrix ret, double c) {
	// Check that the matrix dimensions match
	if (!(A.n == ret.n && A.m == ret.m)) {
		printf("\nERROR: Multiplication failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is properly allocated
	if (A.proper_init) {
		printf("\nERROR: Multiplication failed as the matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const double4_t mplr = {c, c, c, c};
	const int vect_num = A.vects_per_row;
	// Go over the rows of A
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A.n; i++) {
		// Go over each vector on the row
		for (int vect = 0; vect < vect_num; vect++) {
			ret.data[vect_num * i + vect] = A.data[vect_num * i + vect] * mlpr;
		}
	}
	
	return 0;
}


// Function for negating a matrix
// Returns 0 if operation is successful, 1 otherwise
int negate_dense(denseMatrix A, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n == ret.n && A.m == ret.m)) {
		printf("\nERROR: Negation failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || ret.proper_init) {
		printf("\nERROR: Negation failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	if (smult_dense(A, ret, (double)-1.0)) {
		printf("\nERROR: Negation failed as an error occured with scalar multiplication\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}	
	
	return 0;
}


// Function for taking the difference between two matrices (i.e. A - B))
// Returns 0 if operation is successful, 1 otherwise
int diff_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n == B.n && A.n == ret.n && A.m == B.m && A.m == ret.m)) {
		printf("\nERROR: Difference failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || B.proper_init || ret.proper_init) {
		printf("\nERROR: Difference failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Negate B so it can be summed with A
	if (negate_dense(B, ret)) {
		printf("\nERROR: Negation step of difference failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Sum with A
	if (sum_dense(A, ret, ret)) {
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
	if (!(A.n == ret.m && A.m == ret.n)) {
		printf("\nERROR: Transpose failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || ret.proper_init) {
		printf("\nERROR: Transpose failed as some matrix isn't properly allocated\n");
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
				
				ret.data[vect_num0 * j + vect0][elem0] = tmp[elem];
			}
		}
	}
	
	return 0;
}


// Function for computing the Hadamard product (element-wise product A.*B)
// of two denseMatrices
// Returns 0 if operation is successful 1 otherwise
int hprod_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	
}


// Function for computing the Hadamard division (element-wise division A./B)
// of two denseMatrices
// Returns 0 if operation is successful 1 otherwise
int hdiv_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	
}


// Function for computing the dot product between two vectors
// Returns 0 if operation is successful 1 otherwise
int dot_dense(denseMatrix v, denseMatrix u, double* ret) {
	
}


// Function for multiplying two matrices (i.e AB)
// Returns 0 if operation is successful 1 otherwise
int mult_dense(denseMatrix A, denseMatrix B, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.m == B.n && A.n == ret.n && B.m == ret.m)) {
		printf("\nERROR: Multiplication failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || B.proper_init || ret.proper_init) {
		printf("\nERROR: Multiplication failed as some matrix isn't properly allocated\n");
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
	
	const int vect_num = A.vects_per_row;
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


// (Naive) Function for matrix powers
int pow_dense(denseMatrix A, denseMatrix ret, int k) {
	// Check that the matrix dimensions match
	if (!(A.n == ret.n && A.m == ret.m)) {
		printf("\nERROR: Matrix power failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is symmetric
	if (!(A.n == A.m)) {
		printf("\nERROR: Matrix power failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || ret.proper_init) {
		printf("\nERROR: Matrix power failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Create an identity matrix of correct size which will used to compute the power
	denseMatrix I = eye_dense(A.n, A.n);
	// Check that the operation was successful
	if (I.proper_init) {
		printf("\nERROR: Matrix power failed as generating an identity matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	 
	// Multiply A with itself k times in a loop
	for (int i = 0; i < k; i++) {
		if (mult_dense(A, I, I)) {
			printf("\nERROR: Matrix power failed as matrix multiplication failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}
	}
	
	// Copy the contents of I to ret
	if (_copy_dense(ret, I)) {
		printf("\nERROR: Matrix power failed as copying the contents to ret failed\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Free the temporary matrix
	free_denseMatrix(I);
	
	return 0;
}  


// Function for computing the determinant of a matrix
int det_dense(denseMatrix A, double* ret) {
	
}


// ADVANCED MATH OPERATIONS

// Function for inverting a matrix using Gauss-Jordan method
// Modified from: https://rosettacode.org/wiki/Gauss-Jordan_matrix_inversion#C
// Returns 0 if operation is successful 1 otherwise
int inv_dense(denseMatrix A, denseMatrix ret) {
	// Check that the matrix dimensions match
	if (!(A.n == ret.n && A.m == ret.m)) {
		printf("\nERROR: Inversion failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is symmetric
	if (!(A.n == A.m)) {
		printf("\nERROR: Inversion failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || ret.proper_init) {
		printf("\nERROR: Inversion failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Create an identity matrix of correct size which will converted to the inverse
	denseMatrix I = eye_dense(A.n, A.n);
	// Check that the operation was successful
	if (I.proper_init) {
		printf("\nERROR: Inversion failed as generating an identity matrix failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy A create a copy of it
	denseMatrix _A = alloc_denseMatrix(A.n, A.n);
	if (_A.proper_init) {
		printf("\nERROR: Inversion failed as allocating memory for a copy of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Copy the contents of A into _A
	if (_copy_dense(_A, A)) {
		printf("\nERROR: Inversion failed as copying of the contents of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the Frobenius norm of _A
	double4_t frob_vect = {(double)0.0, (double)0.0, (double)0.0, (double)0.0};
	int vect_num = _A.vects_per_row;
	// Go over the rows
	for (int i = 0; i < _A.n; i++) {
		// Go over vectors per row
		for (int vect = 0; vect < vect_num; vect++) {
			frob_vect += _A.data[vect_num * i + vect] * _A.data[vect_num * i + vect];
		}
	}
	double frob = sqrt(frob_vect[0] + frob_vect[1] + frob_vect[2] + frob_vect[3]);
	
	// Define the tolerance with the Frobenius norm and machine epsilon
	double tol = frob * DBL_EPSILON;
	
	// Main loop body
	for (int k = 0; k < _A.n; k++) {
		// Find the pivot
		int vect0 = k / DOUBLE_ELEMS;
		int elem0 = k % DOUBLE_ELEMS;
		double pivot = fabs(_A.data[vect_num * k + vect0][elem0]);
		int pivot_i = k;
		
		// Go over the remaining rows and check if there is a greater pivot
		#pragma omp parallel for schedule(dynamic, 1)
		for (int i = k + 1; i < n; i++) {
			double opt_pivot = fabs(_A.data[vect_num * i + vect0][elem0]);
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
			return 1;
		}
		
		// If the best pivot is not found on row k swap the rows in both 
		// matrix _A aswell as in the matrix I
		if (pivot_i != k) {
			// Swap rows in _A
			#pragma omp parallel for schedule(dynamic, 1)
			for (int vect = vect0; vect < vect_num) {
				double4_t tmp = _A.data[vect_num * k + vect];
				_A.data[vect_num * k + vect] = _A.data[vect_num * pivot_i + vect];
				_A.data[vect_num * pivot_i + vect] = tmp;
			}
			// Swap rows in I
			#pragma omp parallel for schedule(dynamic, 1)
			for (int vect = 0; vect < vect_num) {
				double4_t tmp = I.data[vect_num * k + vect];
				I.data[vect_num * k + vect] = I.data[vect_num * pivot_i + vect];
				I.data[vect_num * pivot_i + vect] = tmp;
			}
		}
		
		// Scale the rows of both _A and I so that the pivot is 1
		pivot = (double)1.0 / pivot;
		double4_t pivot_vect = {pivot, pivot, pivot, pivot};
		// Scale _A
		#pragma omp parallel for schedule(dynamic, 1)
		for (int vect = vect0; vect < vect_num; vect++) {
			_A.data[vect_num * k + vect] *= pivot_vect;
		}
		// Scale I
		#pragma omp parallel for schedule(dynamic, 1)
		for (int vect = 0; vect < vect_num; vect++) {
			I.data[vect_num * k + vect] *= pivot_vect;
		}
		
		// Subtract to get the wanted zeros in _A
		// Go over the rows
		#pragma omp parallel for schedule(dynamic, 1)
		for (int i = 0; i < _A.n; i++) {
			if (i == k) continue;
			double row_pivot = _A.data[i * vect_num + vect0][elem0];
			double4_t row_pivot_vect = {row_pivot, row_pivot, row_pivot, row_pivot};
			// Go over each vector on the row for _A
			for (int vect = vect0; vect < vect_num; vect++) {
				_A.data[vect_num * i + vect] -= _A.data[vect_num * k + vect] * row_pivot_vect;
			}
			// Go over each vector on the row for I
			for (int vect = 0; vect < vect_num; vect++) {
				I.data[vect_num * i + vect] -= I.data[vect_num * k + vect] * row_pivot_vect;
			}		
		}	
	}
	
	// Copy the final inverse from I to ret
	if (_copy_dense(ret, I)) {
		printf("\nERROR: Inversion failed as copying the contents to ret failed\n")
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Free the allocated temporary memory
	free_denseMatrix(I);
	free_denseMatrix(_A);
	
	return 0;
}


// Function for Cholensky decomposition A = LL^T. Works only for s.p.d matrices
// Returns 1 if operation is successful 0 otherwise
int chol_dense(denseMatrix A, denseMatrix L) {
	// Check that the matrix dimensions match
	if (!(A.n == L.n && A.m == L.m)) {
		printf("\nERROR: Cholensky failed as matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrix is symmetric
	if (!(A.n == A.m)) {
		printf("\nERROR: Cholensky failed as the matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A.proper_init || L.proper_init) {
		printf("\nERROR: Cholensky failed as some matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// As we don't want to destroy A create a copy of it
	denseMatrix _A = alloc_denseMatrix(A.n, A.n);
	if (_A.proper_init) {
		printf("\nERROR: Cholensky failed as allocating memory for a copy of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Copy the contents of A into _A
	if (_copy_dense(_A, A)) {
		printf("\nERROR: Cholensky failed as copying of the contents of A failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int vect_num = _A.vects_per_row;
	// Go over the rows of A
	for (int i = 0; i < _A.n; i++) {
		// Allocate memory for update values for L
		denseMatrix a21_T = alloc_denseMatrix(1, _A.n - (i + 1));
		denseMatrix B = alloc_denseMatrix(_A.n - (i + 1), _A.n - (i + 1));
		denseMatrix l21 = alloc_denseMatrix(_A.n - (i + 1), 1);
		
		// Get the needed subarrays
		double a11;
		int a_success = _apply_dense(_A, &a11, i, i);
		if (a_success && a <= 0) {
			printf("\nERROR: Cholensky failed as the given matrix is not symmetric positive definite\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}
		
		denseMatrix a21 = _subarray_dense(_A, i + 1, _A.n, i, i + 1);
		denseMatrix A22 = _subarray_dense(_A, i + 1, _A.n, i + 1, _A.n);
		// Check that the allocations were successful
		if (A22.proper_init || a21.proper_init || B.proper_init || a21_T.proper_init || l21.proper_init || a_success) {
			printf("\nERROR: Cholensky failed as some subarray allocations failed\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}

		// Compute the update values for L and A
		int T_success = transpose_dense(a21, a21_T);
		int m_success = mult_dense(a21, a21_T, B);
		int s_success = smult_dense(B, B, a11);
		int d_success = diff_dense(A22, B, B);
		double l11 = sqrt(a11);
		int s2_success = smult_dense(a21, l11);
		int s3_success = smult_dense(a21, a21, 1. / l11);
		// Check that the operations were successful
		if (T_success || m_success || s_success || d_success || s2_success || s3_success) {
			printf("\nERROR: Cholensky failed as there was a problem with some math operation\n");
			printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
			return 1;
		}
		
		// Place the values back to the arrays
		int p1_success = _place_dense(L, i, i, l11);
		int p2_success = _place_subarray_dense(L)
		
		
	}
	
	return 0;
}


// Function for PLU decomposition
// Returns 1 if operation is successful 0 otherwise
int PLU_dense(denseMatrix A, denseMatrix P, denseMatrix L, denseMatrix U) {
	
}


// Function for solving a system of linear equations of form Ax = b
// using PLU decomposition of the matrix A
// Returns 1 if operation is successful 0 otherwise
int linsolve_dense(denseMatrix A, denseMatrix x, denseMatrix b) {
	
}


// Function for computing the eigendecomposition (that is A = SES^-1 
// where S has the eigenvectors of A as columns and E has the eigenvalues
// of A on the diagonal) of a given matrix A
// Returns 1 if operation is successful 0 otherwise
int eig_dense(denseMatrix A, denseMatrix S, denseMatrix E, denseMatrix S_inv) {
	
} 


