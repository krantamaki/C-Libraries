#include "declare_dense.h"


// Function for computing the dot product between two column vectors
// Returns 0 if operation is successful 1 otherwise
int dot_dense(denseMatrix* v, denseMatrix* u, double* ret) {
	// Check that the matrix dimensions match
	if (!(u->m == v->m && u->n == v->n)) {
		printf("\nERROR: Matrix dimensions don't match\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 2;
	}
	// Check that the 'matrices' are column vectors
	if (!(v->m == 1)) {
		printf("\nERROR: Matrices are not column vectors\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 5;
	}
	// Check that matrices are properly allocated
	if (v->proper_init || u->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
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
