#include "declare_dense.h"


// Function for printing a matrix. The elements will always be printed
// with precision of 3 decimal points. The 
int print_dense(denseMatrix* A) {
	// Check that matrices are properly allocated
	if (A->proper_init) {
		printf("\nERROR: Matrix not properly allocated\n");
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
