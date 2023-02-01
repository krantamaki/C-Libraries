#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../general.h"
#include "../Linear Algebra/dense_matrix.h"
#include "../Data Structures and Algorithms/searching.h"
#include "statistics.h"


// (UNDER CONSTRUCTION!)
// This is a general statistics library utilizing matrices and associated
// operations from dense_matrix.h in the implementations

// Vast majority of the algorithms are applied from the course material
// of course MS-C1620 Statistical Inference provided by Aalto Univerity


// USEFUL HELPER FUNCTIONS

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


// Function that finds the largest element in the matrix
// Uses a brute force search
// Returns 0 if operation is successful and 1 otherwise
int max(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	double best_found = -DBL_MAX;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			if (val > best_found) best_found = val;
		}
	}
	
	// Store the found value
	*ret = best_found;
}


// Function that finds the smallest element in the matrix
// Uses a brute force search
// Returns 0 if operation is successful and 1 otherwise
int min(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	double best_found = DBL_MAX;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			if (val < best_found) best_found = val;
		}
	}
	
	// Store the found value
	*ret = best_found;
}


// DESCRIPTIVE STATISTICS


// TODO: BETTER DESCRIPTIONS


// Finds the median of the values found in the matrix
// Uses quickselect algorithm in the search
// Returns 0 if operation is successful and 1 otherwise
int median(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Turn the denseMatrix into a double array
	double* data = conv_to_arr(A);
	int n = A->m * A->n;
	
	// If the number of elements is even we will average the two middlemost
	// values otherwise we just take the middle value
	if (n % 2 == 0) {
		int low_i = (int)n / 2;
		int high_i = _ceil(n / 2);
		
		quickselect(data, n, low_i, sizeof(double), double_cmp_approx);
		double low = data[low_i];
		
		quickselect(data, n, high_i, sizeof(double), double_cmp_approx);
		double high = data[high_i];
		
		*ret = (low + high) / 2;
	}
	else {
		int mid = (int)n / 2;
		quickselect(data, n, mid, sizeof(double), double_cmp_approx);
		*ret = data[low_i];
	}
	
	// Free temporary array
	free(data);
}


// Finds the b quantile of values found in the matrix
// Uses quickselect algorithm in the search
// Returns 0 if operation is successful and 1 otherwise
int b_quantile(denseMatrix* A, double b, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the given b is valid
	if (!(0.0 < b < 1.0)) {
		printf("\nERROR: Given b not in needed range 0 < b < 1\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Turn the denseMatrix into a double array
	double* data = conv_to_arr(A);
	int n = A->m * A->n;
	int b_i = (int)b * n;
	
	quickselect(data, n, b_i, sizeof(double), double_cmp_approx);
	*ret = data[b_i];
}


// Function for computing the sample mean for given data. 
// Computes the mean across all elements independent of the shape
// of the matrix
// Returns 0 if operation is successful and 1 otherwise
int mean(denseMatrix* A, double* ret) {
	// Check that vector is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int size = A->n * A->m;
	double sum = 0.0;
	const int vect_num = A->vects_per_row;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		double4_t sum_vect = {0.0, 0.0, 0.0, 0.0};
		// Go over the columns
		for (int vect = 0; vect < vect_num; vect++) {
			sum_vect += A->data[vect_num * i + vect];
		}
		sum += sum_vect[0] + sum_vect[1] + sum_vect[2] + sum_vect[3];
	}
	
	// Store the found mean
	*ret = sum / size;
}


// TODO: Mode
//int mode(denseMatrix* A, double* ret) {}


// Function for computing the weighted mean for given data.
// Computes the weighted mean across all elements independent of the shape
// or the matrix.
// Assumes that the given weights are proper i.e. 0 < W_ij < 1 and 
// the weights sum to 1.
// Returns 0 if operation is successful and 1 otherwise 
int weighted_mean(denseMatrix* A, denseMatrix* W, double* ret) {
	// Check that the dimensions of the matrices match
	if (!(A->n == W->n && A->m == W->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are properly allocated
	if (A->proper_init || W->proper_init) {
		printf("\nERROR: Matrices not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	int size = A->n * A->m;
	double sum = 0.0;
	const int vect_num = A->vects_per_row;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		double4_t sum_vect = {0.0, 0.0, 0.0, 0.0};
		// Go over the columns
		for (int vect = 0; vect < vect_num; vect++) {
			sum_vect += W->data[vect_num * i + vect] * A->data[vect_num * i + vect];
		}
		sum += sum_vect[0] + sum_vect[1] + sum_vect[2] + sum_vect[3];
	}
	
	// Store the found mean
	*ret = sum / size;
}


// Function that computes the k-th raw sample moment
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int k_moment(denseMatrix* A, int k, double* ret) {
	// Check that matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the mean
	double mean_val;
	mean(A, &mean_val);
	int size = A->n * A->m;
	
	double sum = 0.0;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			sum += pow(val - mean_val, (double)k);
		}
	}
	
	*ret = sum / size;
}


// Function for computing the sample variance for given data.
// Computes the variance across all elements independent of the shape
// of the matrix
// Returns 0 if operation is successful and 1 otherwise
int var(denseMatrix* A, double* ret) {
	// Check that matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the mean
	double mean_val;
	mean(A, &mean_val);
	int size = A->n * A->m;
	
	double sum = 0.0;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			sum += pow(val - mean_val, 2.0);
		}
	}
	
	// Store the variance
	*ret = sum / (size - 1);
}


// Function for computing the sample standard deviation for given data.
// Computes the standard deviation across all elements independent of the 
// shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int sd(denseMatrix* A, double* ret) {
	// Check that vector is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	double variance;
	var(A, &variance);
	*ret = pow(variance, 1/2);
}


// Function for computing the median abslute deviation for given data.
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int mad(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Allocate memory for a secondary matrix to store the elementwise
	// absolute deviations
	denseMatrix* _A = alloc_denseMatrix(A->n, A->m);
	if (_A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		free_denseMatrix(_A);
		return 1;
	}
	
	// Compute the mean
	double mean_val;
	mean(A, %mean_val);
	
	int vect_num = A->vects_per_row;
	// Go over the rows
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < _A->n; i++) {
		// Go over the columns
		for (int j = 0; j < _A->m; j++) {
			double val;
			_apply_dense(A, &val, i, j);
			double abs_dev = fabs(val - mean);
			_place_dense(_A, abs_dev, i, j);
		}
	}
	
	// Find the median of the absolute deviations
	median(_A, &ret);
}


// Function that computes the sample skewness for given data
// Computed across all elements independent of the shape of the matrix
// If skewness factor > 0 the distribution is skewed to the right
// and if skewness factor < 0 the distribution is skewed to the left
// Returns 0 if operation is successful and 1 otherwise
int skewness(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the 3rd moment
	double moment_3;
	k_moment(A, 3, &moment_3);
	
	// Compute the standard deviation
	double sd_val;
	sd(A, &sd_val);
	
	*ret = moment_3 / pow(sd_val, 3.0);
}


// Function that computes the sample kurtosis coefficient
// Computed across all elements independent of the shape of the matrix
// If kurtosis value > 0 the distribution has heavier tails than normal
// distribution and if kurtosis value < 0 lighter tails
// Returns 0 if operation is successful and 1 otherwise
int kurtosis(denseMatrix* A, double* ret) {
	// Check that the matrix is properly allocated
	if (A->proper_init) {
		printf("\nERROR: Given matrix is not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the 4th moment
	double moment_3;
	k_moment(A, 3, &moment_3);
	
	// Compute the standard deviation
	double sd_val;
	sd(A, &sd_val);
	
	*ret = moment_3 / pow(sd_val, 4.0) - 3;
}


// Function that computes the sample covariance of the values in two matrices
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int cov(denseMatrix* A, denseMatrix* B, double* ret) {
	// Check that the dimensions of the matrices match
	if (!(A->n == B->n && A->m == B->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are properly allocated
	if (A->proper_init || B->proper_init) {
		printf("\nERROR: Matrices not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the means
	double mean_A, mean_B;
	mean(A, &mean_A);
	mean(B, &mean_B);
	
	int size = A->n * A->m;
	double sum = 0.0;
	// Go over the rows
	for (int i = 0; i < A->n; i++) {
		// Go over the columns
		for (int j = 0; j < A->m; j++) {
			double a_ij, b_ij;
			_apply_dense(A, &a_ij, i, j);
			_apply_dense(B, &b_ij, i, j);
			sum += (a_ij - mean_A) * (b_ij - mean_B);
		}
	}
	
	*ret = sum / (size - 1);
}


// Function that computes the sample correlation of the values in two matrices
// Computed across all elements independent of the shape of the matrix
// Returns 0 if operation is successful and 1 otherwise
int corr(denseMatrix* A, denseMatrix* B, double* ret) {
	// Check that the dimensions of the matrices match
	if (!(A->n == B->n && A->m == B->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that the matrices are properly allocated
	if (A->proper_init || B->proper_init) {
		printf("\nERROR: Matrices not properly allocated\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	
	// Compute the standard deviations for the matrices
	double sd_A, sd_B;
	sd(A, &sd_A);
	sd(B, &sd_B);
	
	// Compute the covariance between the matrices
	double cov_AB;
	cov(A, B, &cov_AB);
	
	*ret = cov_AB / (sd_A * sd_B);
}

