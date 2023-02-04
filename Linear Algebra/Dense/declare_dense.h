#ifndef LINEAR_ALGEBRA_DENSE
#define LINEAR_ALGEBRA_DENSE

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../../general.h"


// (UNDER CONSTRUCTION!)
// This is a general linear algebra library using dense matrices 
// with double precision floating point numbers.
// Optimized to allow multithreading, vectorization (256 bit SIMD) and ILP, 
// but lacks prefetching, improved cache management and proper register reuse

// Unless otherwise mentioned the algorithms are based on the material
// from the course MS-E1651 Numerical Matrix Computation notes by Antti 
// Hannukainen or from the book Numerical Linear Algebra by Trefethen and Bau


// Matrix struct
typedef struct {
	int n;  // Number of rows
	int m;  // Number of columns
	int proper_init;  // 0 if proper, 1 otherwise
	size_t vects_per_row;
	double4_t* data;
} denseMatrix;


// Visualization
#define MAX_PRINTS 5  // Maximum number of rows printed and values per row printed
int print_dense(denseMatrix* A);
void _print_all(denseMatrix* A);  // For debugging


// General functions
denseMatrix* alloc_denseMatrix(const int n, const int m);
denseMatrix* eye_dense(const int n, const int m);
denseMatrix* conv_to_denseMatrix(double* arr, const int n, const int m);
double* conv_to_arr(denseMatrix* A);
void free_denseMatrix(denseMatrix* A);
int _apply_dense(denseMatrix* A, double* ret, const int i, const int j);
int _place_dense(denseMatrix* A, const double val, const int i, const int j);
denseMatrix* _subarray_dense(denseMatrix* A, const int n_start, const int n_end, const int m_start, const int m_end);
int _place_subarray_dense(denseMatrix* A, denseMatrix* B, const int n_start, const int n_end, const int m_start, const int m_end);
int init_eye_dense(denseMatrix* ret);
int copy_dense(denseMatrix* dst, denseMatrix* src);


// Basic math operations
int sum_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret);
int diff_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret);
int smult_dense(denseMatrix* A, denseMatrix* ret, const double c);
int negate_dense(denseMatrix* A, denseMatrix* ret);
int transpose_dense(denseMatrix* A, denseMatrix* ret);
int hprod_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret);
int hdiv_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret);
int hpow_dense(denseMatrix* A, denseMatrix* ret, double k);
int dot_dense(denseMatrix* v, denseMatrix* u, double* ret);
int mult_dense(denseMatrix* A, denseMatrix* B, denseMatrix* ret);
int pow_dense(denseMatrix* A, denseMatrix* ret, const int k);

// TODO:
//int trace_dense(denseMatrix* A, double* ret);
//int row_ech_dense(denseMatrix* A);
//int det_dense(denseMatrix* A, double* ret);


// Advanced math operations
int inv_dense(denseMatrix* A, denseMatrix* ret);
int chol_dense(denseMatrix* A, denseMatrix* L);
int PLU_dense(denseMatrix* A, denseMatrix* P, denseMatrix* L, denseMatrix* U);
int trilsolve_dense(denseMatrix* L, denseMatrix* x, denseMatrix* b);
int triusolve_dense(denseMatrix* U, denseMatrix* x, denseMatrix* b);
int linsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b);

// TODO:
//int cgsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b);
//int cholsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b);
//int invsolve_dense(denseMatrix* A, denseMatrix* x, denseMatrix* b);
//int eig_dense(denseMatrix* A, denseMatrix* S, denseMatrix* E, denseMatrix* S_inv);
//int schur_dense(denseMatrix* A, denseMatrix* Q, denseMatrix* U);
//int exp_dense(denseMatrix* A, denseMatrix* ret);


#endif
