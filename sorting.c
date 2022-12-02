#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

// The comparison functions for datatypes int, float, double and char*.
// Comparison functions are passed to the sorting algorithms and must
// correspond with the datatype of the elements of the array

// Comparison function for 32-bit signed integers
int int_cmp(const int* int1, const int* int2, const int desc) {
	return desc == 1 ? int1 - int2 : int2 - int1;
}

// Comparison function for 32-bit floating point numbers
int float_cmp(const float* float1, const float* float2, const int desc) {
	int cmp;
	if (float1 == float2) cmp = 0;
	else if (float1 < float2) cmp = -1;
	else cmp = 1;
	return desc == 1 ? cmp : -cmp;
}

// Comparison function for 64-bit floating point numbers
int double_cmp(const double* double1, const double* double2, const int desc) {
	int cmp;
	if (double1 == double2) cmp = 0;
	else if (double1 < double2) cmp = -1;
	else cmp = 1;
	return desc == 1 ? cmp : -cmp;
}

// Comparison function for NULL TERMINATED char arrays
// Differs from string.h libraries strcmp() function by comparing
// the sum of the byte representations of the chars rather that finding 
// the first different char and comparing that
int str_cmp(const char* str1, const char* str2, int desc) {
	int sum1 = 0, sum2 = 0, i = 1;
	char str_char = str1[0];

	while (str_char != '\0') {
		sum1 += (int) str_char;
		str_char = str1[i];
		i++;
	}

	i = 1;
	char str_char = str2[0];

	while (str_char != '\0') {
		sum2 += (int) str_char;
		str_char = str2[i];
		i++;
	}

	return int_cmp(sum1, sum2, desc);
}

// Standard insertion sort algorithm. This is defined first as it is what
// quick and merge sort algorithms will default to with small enough array
// sizes. Best case time complexity O(n) and worst case O(n^2) so not 
// recommended for larger arrays.


