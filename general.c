#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "general.h"


// This file contains general functions that multiple different 
// files in this folder have dependancy on. Should always be included
// when compiling the other files or any code that depends on them.


// VOID POINTED ARRAY FUNCTIONS

// Function for getting the ith element of a void pointer array
void* _apply(void* arr, const int i, size_t size) {
	return (void*)((char*)arr + (int)size * i);
}

// Function for placing the value of src into the pointer of dst
void _place(void* dst, void* src, size_t size) {
	unsigned char *ptr1 = dst, *ptr2 = src;
	for (size_t i = 0; i < size; i++) {
		ptr1[i] = ptr2[i];
	}
}

// Function for swapping the references of two pointers
void _swap(void* elem1, void* elem2, size_t size) {
	unsigned char *ptr1 = elem1, *ptr2 = elem2, temp;
	for (size_t i = 0; i < size; i++) {
		temp = ptr1[i];
		ptr1[i] = ptr2[i];
		ptr2[i] = temp;
	}
}


// BIT OPERATIONS

// Returns the bit at index i of the given byte
int get_bit(char byte, int i) {
	return (byte >> i) & 1;
}

// Returns the byte with bit at index i toggled
char toggle_bit(char byte, int i) {
	byte ^= 1 << i;
	return byte;
}


// COMPARISON FUNCTIONS

// Comparison functions are passed to the sorting algorithms and must
// correspond with the datatype of the elements of the array

// Comparison function for 32-bit signed integers
int int_cmp(void* ptr1, void* ptr2) {
	int int1 = *(const int*)ptr1;
	int int2 = *(const int*)ptr2;
	return int1 - int2;
}

// Comparison function for 32-bit floating point numbers
int float_cmp(void* ptr1, void* ptr2) {
	float float1 = *(const float*)ptr1;
	float float2 = *(const float*)ptr2;
	
	int cmp;
	if (float1 == float2) cmp = 0;
	else if (float1 < float2) cmp = -1;
	else cmp = 1;
	return cmp;
}

// Comparison function for 64-bit floating point numbers
// Naive implementation
int double_cmp(void* ptr1, void* ptr2) {
	double double1 = *(const double*)ptr1;
	double double2 = *(const double*)ptr2;
	
	int cmp;
	if (double1 == double2) cmp = 0;
	else if (double1 < double2) cmp = -1;
	else cmp = 1;
	return cmp;
}

// Comparison function for 64-bit floating point numbers
// Approximate implementation
int double_cmp_approx(void* ptr1, void* ptr2) {
	double double1 = *(const double*)ptr1;
	double double2 = *(const double*)ptr2;
	
	double avg = (double1 + double2) / 2;
	
	int cmp;
	if (fabs(double1 - double2) < DBL_EPSILON * avg) cmp = 0;
	else if (double1 < double2) cmp = -1;
	else cmp = 1;
	return cmp;
}

// Comparison function for NULL TERMINATED char arrays
// Differs from string.h libraries strcmp() function by comparing
// the sum of the byte representations of the chars rather that finding 
// the first different chars and comparing them
int str_cmp(void* ptr1, void* ptr2) {
	char* str1 = (char*)ptr1;
	char* str2 = (char*)ptr2;
	
	int sum1 = 0, sum2 = 0, i = 1;
	char str_char = str1[0];

	while (str_char != '\0') {
		sum1 += (int) str_char;
		str_char = str1[i];
		i++;
	}

	i = 1;
	str_char = str2[0];

	while (str_char != '\0') {
		sum2 += (int) str_char;
		str_char = str2[i];
		i++;
	}

	return sum1 - sum2;
}


// USEFUL MATH FUNCTIONS

// Function for computing the power for a given integer
int _pow(int x, int n) {
    int ret = 1;
    for (int i = 0; i < n; i++) {
        ret *= x;
    }

    return ret;
}

// Function for "dividing up" two integers
int _ceil(int a, int b) {
    return (a + b - 1) / b;
}


// FUNCTIONS FOR GENERATING RANDOM ARRAYS

// Function for generating a wanted sized array of random 32-bit integers
// All elements have values in range [0, upper]
// Assumes that enough memory has already been allocated
void rand_int_arr(int* arr, const int n, const int upper) {
	for (int i = 0; i < n; i++) {
		arr[i] = rand() % upper;
	}
}

// Function for generating a wanted sized array of random 32-bit floats
// All elements have values in range [0.0, upper]
// NOTE! The values don't follow uniform distribution due to the 
// properties of floating point numbers, but for testing purposes
// this is considered adequate.
// Assumes that enough memory has already been allocated
void rand_float_arr(float* arr, const int n, const float upper) {
	for (int i = 0; i < n; i++) {
		arr[i] = ((float)rand() / (float)RAND_MAX) * upper;
	}
}

// Function for generating a wanted sized array of random 64-bit floats
// All elements have values in range [0.0, upper]
// NOTE! The values don't follow uniform distribution due to the 
// properties of floating point numbers, but for testing purposes
// this is considered adequate.
// Assumes that enough memory has already been allocated
void rand_double_arr(double* arr, const int n, const float upper) {
	for (int i = 0; i < n; i++) {
		arr[i] = ((double)rand() / (double)RAND_MAX) * upper;
	}
}


