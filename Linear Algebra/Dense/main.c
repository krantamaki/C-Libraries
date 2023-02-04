#include "declare_dense.h"


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
// - compile (using the provided Makefile): make
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
	
	return 0;
}
