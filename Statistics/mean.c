#include "declare_stats.h"


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
