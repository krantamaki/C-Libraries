#include "declare_dense.h"


// Function for solving a system of linear equations of form Ax = b
// using PLU decomposition of the matrix A
// Returns 0 if operation is successful 1 otherwise
int linsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b) {
	// Check that the matrix dimensions match
	if (!(A->m == x->n && A->n == b->n && x->m == b->m)) {
		printf("\nERROR: Matrix dimensions don't match\n");
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
		printf("\nERROR: Matrix isn't symmetric\n");
		printf("FOUND: In file %s at function %s on line %d\n", __FILE__, __func__, __LINE__);
		return 1;
	}
	// Check that matrices are properly allocated
	if (A->proper_init || x->proper_init || b->proper_init) {
		printf("\nERROR: Given matrix isn't properly allocated\n");
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
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
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
	copy_dense(_b, b);
	if (_b->proper_init) {
		printf("\nERROR: Memory allocation for a temporary matrix failed\n");
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
	transpose_dense(P, P_T);
	mult_dense(P_T, _b, _b);
	if (trilsolve_dense(L, y, _b)) {
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
	mult_dense(P, x, x);
	
	// Free allocated memory
	free_denseMatrix(P);
	free_denseMatrix(P_T);
	free_denseMatrix(L);
	free_denseMatrix(U);
	free_denseMatrix(y);
	free_denseMatrix(_b);
	
	return 0;
}
