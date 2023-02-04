#include "declare_stats.h"


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
