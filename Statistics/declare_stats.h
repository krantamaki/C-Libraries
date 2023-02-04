#ifndef STATISTICS
#define STATISTICS

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../general.h"
#include "../Linear Algebra/Dense/declare_dense.h"
#include "../Data Structures and Algorithms/searching.h"


// (UNDER CONSTRUCTION!)
// This is a general statistics library utilizing matrices and associated
// operations from dense_matrix.h in the implementations

// Vast majority of the algorithms are applied from the course material
// of course MS-C1620 Statistical Inference provided by Aalto Univerity


// Apply functions
int apply_rowfunc(denseMatrix* A, denseMatrix* ret, int (*func)(denseMatrix*, double*));
int apply_colfunc(denseMatrix* A, denseMatrix* ret, int (*func)(denseMatrix*, double*));


// Descriptive statistics
int max(denseMatrix* A, double* ret);
int min(denseMatrix* A, double* ret);
int median(denseMatrix* A, double* ret);
int b_quantile(denseMatrix* A, double b, double* ret);
int mean(denseMatrix* A, double* ret);
int weighted_mean(denseMatrix* A, denseMatrix* W, double* ret);
int k_moment(denseMatrix* A, int k, double* ret);
int var(denseMatrix* A, double* ret);
int sd(denseMatrix* A, double* ret);
int mad(denseMatrix* A, double* ret);
int skewness(denseMatrix* A, double* ret);
int kurtosis(denseMatrix* A, double* ret);
int cov(denseMatrix* A, denseMatrix* B, double* ret);
int corr(denseMatrix* A, denseMatrix* B, double* ret);

// TODO:
//int mode(denseMatrix* A, double* ret);


#endif
