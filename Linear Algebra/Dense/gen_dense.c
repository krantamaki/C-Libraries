#include "declare_dense.h"


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
		printf("\nERROR: Memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		
		// Return a matrix which signifies failure
		ret->n = n;
		ret->m = m;
		ret->vects_per_row = _ceil(m, DOUBLE_ELEMS);;
		ret->data = NULL;
		ret->proper_init = 1;
		return ret;
	}
	
	const int lower_dim = n >= m ? m : n;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < lower_dim; i++) {
		_place_dense(ret, (double)1.0, i, i);
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
		printf("\nERROR: Memory allocation failed\n");
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


// Function for converting a denseMatrix to a double array
// Returns NULL if operation failse
double* conv_to_arr(denseMatrix* A) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return NULL;
	}
	
	// Allocate enough memory
	double* ret = (double*)malloc(A->n * A->m * sizeof(double));
	
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			ret[i * A->n + j] = val;
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
// Returns 0 if operation is successful, 1 otherwise
int _apply_dense(denseMatrix* A, double* ret, const int i, const int j) {
	// Check that the indexes are within proper range
	if (!(i >= 0 && i < A->n && j >= 0 && j < A->m)) {
		printf("\nERROR: Given indeces exceed the dimensions of the matrix\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	// Check that A is properly initialized
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
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
// Returns 0 if operation is successful, 1 otherwise
int _place_dense(denseMatrix* A, const double val, const int i, const int j) {
	// Check that the wanted index is viable for the matrix
	if (!(i < A->n && j < A->m)) {
		printf("\nERROR: Given indeces exceed the dimensions of the matrix\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
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
		printf("\nERROR: Given matrix is not properly allocated\n");
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
		printf("\nERROR: Subarray memory allocation failed\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return ret;
	}
	
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i0 = 0; i0 < n_end - n_start; i0++) {
		int i = i0 + n_start;
		// Go over the columns
		for (int j0 = 0; j0 < m_end - m_start; j0++) {
			int j = j0 + m_start;
			double val;
			_apply_dense(A, &val, i, j);
			_place_dense(ret, val, i0, j0);
		}
	}
	
	return ret;
}


// (Naive) Function for placing an denseMatrix B into a wanted position in another denseMatrix A
// Returns 0 if operation is successful, 1 otherwise
int _place_subarray_dense(denseMatrix* A, denseMatrix* B, const int n_start, const int n_end, const int m_start, const int m_end) {
	// Check that the matrix is properly allocated
	if (A->proper_init || B->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the dimensions are proper for A
	if (!(n_start < n_end && n_end <= A->n && m_start < m_end && m_end <= A->m)) {
		printf("\nERROR: The start index of a slice must be smaller than end index and end index must be equal or smaller than matrix dimension");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	// Check that the dimensinos are proper for B
	if (!(n_end - n_start == B->n && m_end - m_start == B->m)) {
		printf("\nERROR: Size of the slice must correspond with the dimensions of the placed subarray\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 3;
	}
	
	// Go over the row values
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i0 = 0; i0 < n_end - n_start; i0++) {
		int i = i0 + n_start;
		// Go over the column values
		for (int j0 = 0; j0 < m_end - m_start; j0++) {
			int j = j0 + m_start;
			double B_ij;
			_apply_dense(B, &B_ij, i0, j0);
			_place_dense(A, B_ij, i, j);
		}
	}
	
	return 0;
}


// Function for initializing an allocated denseMatrix as an identity matrix
// NOTE: Only adds the ones on the diagonal.
// Returns 0 if operation is successful, 1 otherwise
int init_eye_dense(denseMatrix* ret) {
	// Check that matrices are properly allocated
	if (ret->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	const int lower_dim = ret->n >= ret->m ? ret->m : ret->n;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < lower_dim; i++) {
		_place_dense(ret, (double)1.0, i, i);
	}

	return 0;
}


// Function for copying the values of one matrix (src) into another (dst)
// NOTE! Could be changed to memcpy implementation (although that would 
// probably be less eficient)
// Returns 0 if operation is successful, 1 otherwise
int copy_dense(denseMatrix* dst, denseMatrix* src) {
	// Check that the matrix dimensions match
	if (!(dst->n == src->n && dst->m == src->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that matrices are properly allocated
	if (dst->proper_init || src->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
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


