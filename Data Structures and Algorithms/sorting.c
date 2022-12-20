#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "../general.h"
#include "sorting.h"


// INSERTION SORT

// Insertion sort algorithm for sorting subarray arr[start, end]. 
// This is defined first as it is what quick- and mergesort algorithms 
// will default to with small enough array sizes. 
// Space complexity of O(1)
// Best case time complexity O(n) and worst case O(n^2) so not 
// recommended for larger arrays. NOTE! Not parallelized.
void _insertion_sort(void *arr, const int start, const int end, 
					 size_t size, int (*cmp)(void*, void*)) {
	for (int i = start + 1; i < end; i++) {
		int j = i;
		while (j > start && (*cmp)(_apply(arr, j - 1, size), _apply(arr, j, size)) > 0) {
			_swap(_apply(arr, j - 1, size), _apply(arr, j, size), size);
			j--;
		}
	}
}

// Function wrapper for calling above defined _insertion_sort for
// the whole array. 
void insertion_sort(void *arr, const int n, size_t size,
					int (*cmp)(void*, void*)) {
	if (n <= 1) return;
	_insertion_sort(arr, 0, n, size, cmp);
}


// MERGESORT

// Function for merging two sub arrays (arr[start:mid + 1] and arr[mid + 1: end])
// in the correct order
void _merge(void* arr, const int start, const int mid, const int end,
			size_t size, int (*cmp)(void*, void*)) {
	int len_left = (mid + 1) - start;
	int len_right = end - mid;
	// Allocate temporary arrays
	void* left = (void*)malloc(len_left * size);
	void* right = (void*)malloc(len_right * size);
	
	// Copy the data to the temp arrays
	for (int i = 0; i < len_left; i++) {
		_place(_apply(left, i, size), _apply(arr, start + i, size), size);
	}
	for (int i = 0; i < len_right; i++) {
		_place(_apply(right, i, size), _apply(arr, (mid + 1) + i, size), size);
	}
	
	// Merge the temporary arrays to arr
	int left_i = 0, right_i = 0, arr_i = start;
	while (left_i < len_left && right_i < len_right) {
		if ((*cmp)(_apply(left, left_i, size), _apply(right, right_i, size)) > 0) {
			_place(_apply(arr, arr_i, size), _apply(right, right_i, size), size);
			right_i++;
		}
		else {
			_place(_apply(arr, arr_i, size), _apply(left, left_i, size), size);
			left_i++;
		}
		arr_i++;
	}
	
	// Copy remaining elements
	for (int i = left_i; i < len_left; i++) {
		_place(_apply(arr, arr_i, size), _apply(left, i, size), size);
		arr_i++;
	}
	for (int i = right_i; i < len_right; i++) {
		_place(_apply(arr, arr_i, size), _apply(right, i, size), size);
		arr_i++;
	}
	
	// Free allocated memory
	free(left);
	free(right);
}

// Recursively called assisting function for mergesort
void _mergesort(void* arr, const int start, const int end, 
				size_t size, int (*cmp)(void*, void*)) {
	if (start < end) {  // Sanity check (should always be the case)
		if (end - start > THRESHOLD) {
			int mid = (end + start) / 2;
			#pragma omp taskgroup  // Start multiple recursive tasks in parallel
			{
				#pragma omp task shared(arr)  // Tasks share the same array
				_mergesort(arr, start, mid + 1, size, cmp);
				#pragma omp task shared(arr)
				_mergesort(arr, mid + 1, end, size, cmp);
			}
			// Merge the sorted subarrays
			_merge(arr, start, mid, end, size, cmp);
		}
		// If the arrays are small enough just use insertion sort
		else _insertion_sort(arr, start, end, size, cmp);
	}
}

// The main mergesort function
// As this is not an in-place implementation the space complexity is O(n)
// Time complexity is O(n*log_2(n)). Parallelized
void mergesort(void* arr, const int n, size_t size,
			   int (*cmp)(void*, void*)) {
	if (n <= 1) return;
	// Initialize parallelizations
	#pragma omp parallel
	#pragma omp single
	_mergesort(arr, 0, n, size, cmp);
}


// QUICKSORT

// Function for three-way partition using random pivot and the
// Dutch National Flag Algorithm
intTuple _partition_3way(void* arr, const int start, const int end, 
					     size_t size, int (*cmp)(void*, void*)) {
	// Handle 2 element case
	if (end - start <= 1) {
		if ((*cmp)(_apply(arr, start, size), _apply(arr, end, size)) > 0) {
			_swap(_apply(arr, start, size), _apply(arr, end, size), size);
		}
		intTuple ret;
		ret._1 = start;
		ret._2 = end;
		return ret;
	}		
	
	// Find a random pivot	
	int rand_i = (rand() % (end - start)) + start;
	_swap(_apply(arr, rand_i, size), _apply(arr, end - 1, size), size);
	void* pivot = _apply(arr, end - 1, size);
	int i = start;
	int mid = start;
	int j = end;
	
	while (mid <= j) {
		if ((*cmp)(_apply(arr, mid, size), pivot) < 0) {
			_swap(_apply(arr, i, size), _apply(arr, mid, size), size);
			i++;
			mid++;
		}
		else if ((*cmp)(_apply(arr, mid, size), pivot) > 0) {
			_swap(_apply(arr, mid, size), _apply(arr, j, size), size);
			j--;
		}
		else mid++;
	}
	
	intTuple ret;
	ret._1 = i - 1;
	ret._2 = j + 1;
	
	return ret;
}

// Recursively called helper function for quicksort
void _quicksort(void* arr, const int start, const int end, 
				size_t size, int (*cmp)(void*, void*)) {
	if (start < end) {  // Sanity check
		if (end - start > THRESHOLD) {
			intTuple pivots = _partition_3way(arr, start, end, size, cmp);
			#pragma omp taskgroup  // Start multiple recursive tasks in parallel
			{
				#pragma omp task shared(arr)  // Tasks share the same array
				_quicksort(arr, start, pivots._1 + 1, size, cmp);
				#pragma omp task shared(arr)
				_quicksort(arr, pivots._2, end, size, cmp);
			}
		}
		// If the arrays are small enough just use insertion sort
		else _insertion_sort(arr, start, end, size, cmp);
	}			
}

// The main quicksort function
// Space complexity O(1) and time complexity O(n*log_2(n)). Parallelized
void quicksort(void* arr, const int n, size_t size,
			   int (*cmp)(void*, void*)) {
	if (n <= 1) return;
	// Initialize parallelizations
	#pragma omp parallel
	#pragma omp single
	_quicksort(arr, 0, n, size, cmp);	   		   
}


// RADIX SORT

// Function for 32-bit signed integer lsd radix sort
// Space complexity O(n) and time complexity O(n)
// (as the number of possible elements 256 is considered insignificant)
// NOTE! Not parallelized (yet?)
// NOTE! The comparison function is not used but is included as parameter
// for compatibility
void int_radix_sort(void* arr, const int n, size_t size,
					int (*cmp)(void*, void*)) {
	if (n <= 1) return;
	
	// Allocate memory for auxiliary arrays
	int* result = (int*)malloc(n * sizeof(int));
	int* temp = (int*)malloc(n * sizeof(int));  // Not actually necessary
	
	// Copy original array to temp array
	memcpy(temp, arr, n * sizeof(int));
	
	// Allocate memory for the 8-bit arrays
	int* count = (int*)malloc(256 * sizeof(int));
	int* start = (int*)malloc(256 * sizeof(int));
	
	// Initialize needed variables
	int index = 0, num = 0;
	
	// Toggle the sign bit for all elements
	for (int i = 0; i < n; i++) temp[i] ^= 1 << 31;
	
	// The 32-bit integer is split into four bytes which are looped over
	for (int byte = 0; byte < 4; byte++) {
		// Zero the byte arrays
		for (int i = 0; i < 256; i++) {
			count[i] = 0;
			start[i] = 0;
		}
		
		// Loop over the array and count up the occurences of each byte
		for (int i = 0; i < n; i++) {
			num = temp[i];
			index = (num >> (8 * byte)) & 0xff;
			count[index]++;
		}
		
		// Gather the cumulative values of each byte
		for (int i = 1; i < 256; i++) {
			start[i] = start[i - 1] + count[i - 1];
		}
		
		// Alter the output array
		for (int i = 0; i < n; i++) {
			num = temp[i];
			index = (num >> (8 * byte)) & 0xff;
			result[start[index]] = num;
			start[index]++;
		}
		
		// Copy the results to temp array
		memcpy(temp, result, n * sizeof(int));
	}
	
	// Toggle the sign bit back
	for (int i = 0; i < n; i++) temp[i] ^= 1 << 31;
	
	// Copy the result to original array
	memcpy(arr, temp, n * sizeof(int));
	
	// Free allocated memory
	free(temp);
	free(result);
	free(count);
	free(start);   
}

// TODO: Implement floating point radix sort


// TESTING FUNCTIONS

// Function that validates that an array is indeed sorted
int validate_sort(void* arr, const int n, size_t size,
				  int (*cmp)(void*, void*)) {
	for (int i = 0; i < n - 1; i++) {
		if ((*cmp)(_apply(arr, i, size), _apply(arr, i, size)) > 0) return 0;
	}
	
	return 1;			 
}


// Function for timing the execution time of a function 'sort'
// As this is used to time sorting algorithms the parameters needed
// to pass to the function 'sort' are also parameters here.
double sort_timer(void (*sort)(void *, const int, size_t, int (*cmp)(void*, void*)),
				  void* arr, const int n, size_t size, int (*cmp)(void*, void*)) {
	clock_t begin = clock();
	(*sort)(arr, n, size, cmp);
	clock_t end = clock();
	
	return (double)(end - begin) / CLOCKS_PER_SEC;
}


/*
// MAIN FUNCTION (ONLY FOR TESTING PURPOSES)
// To compile this: 
// - navigate to C-Libraries folder
// - compile: gcc -fopenmp -Wall "Data Structures and Algorithms/sorting.c" general.c -o sorting.o
// - run: ./sorting.o
int main() {
	time_t t = time(NULL);
	srand((unsigned) t);
	// Test each algorithm with random array of 10 integers and print
	// the unsorted and sorted array
	printf("Testing each algorithm with random array of 10 ints:\n");
	int* small_arr = (int*)malloc(10 * sizeof(int));
	rand_int_arr(small_arr, 10, 10);
	printf("\nThe unsorted array is:\n");
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_arr[i]);
	}
	
	// Copy the array to temporary array for sorting
	int* small_sort_arr = (int*)malloc(10 * sizeof(int));
	memcpy(small_sort_arr, small_arr, 10 * sizeof(int));
	
	// Insertion sort
	printf("\nSorted using insertion sort:\n");
	insertion_sort(small_sort_arr, 10, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, 10 * sizeof(int));
	
	// Mergesort
	printf("\nSorted using mergesort:\n");
	mergesort(small_sort_arr, 10, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, 10 * sizeof(int));
	
	// Quicksort
	printf("\nSorted using quicksort:\n");
	quicksort(small_sort_arr, 10, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, 10 * sizeof(int));
	
	// Radix sort
	printf("\nSorted using radix sort:\n");
	int_radix_sort(small_sort_arr, 10, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, 10 * sizeof(int));
	
	// Test each algorithm using random array of 10 000 integers
	// and time them
	printf("\n\nTesting all algorithms with an array of 10 000 ints:\n");
	int* big_arr = (int*)malloc(10000 * sizeof(int));
	rand_int_arr(big_arr, 10000, 100);
	int* big_sort_arr = (int*)malloc(10000 * sizeof(int));
	memcpy(big_sort_arr, big_arr, 10000 * sizeof(int));
	
	// Insertion sort
	printf("\nTesting insertion sort:\n");
	double exc_time = sort_timer(insertion_sort, big_sort_arr, 10000, sizeof(int), int_cmp);
	int valid = validate_sort(big_sort_arr, 10000, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	memcpy(big_sort_arr, big_arr, 10000 * sizeof(int));
	
	// Merge sort
	printf("\nTesting mergesort:\n");
	exc_time = sort_timer(mergesort, big_sort_arr, 10000, sizeof(int), int_cmp);
	valid = validate_sort(big_sort_arr, 10000, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	memcpy(big_sort_arr, big_arr, 10000 * sizeof(int));

	// Quicksort
	printf("\nTesting quicksort:\n");
	exc_time = sort_timer(quicksort, big_sort_arr, 10000, sizeof(int), int_cmp);
	valid = validate_sort(big_sort_arr, 10000, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	memcpy(big_sort_arr, big_arr, 10000 * sizeof(int));
	
	// Radix sort
	printf("\nTesting radix sort:\n");
	exc_time = sort_timer(int_radix_sort, big_sort_arr, 10000, sizeof(int), int_cmp);
	valid = validate_sort(big_sort_arr, 10000, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 1000.0);
	}
	memcpy(big_sort_arr, big_arr, 10000 * sizeof(int));
	
	// Free allocated memory
	free(small_arr);
	free(small_sort_arr),
	free(big_arr);
	free(big_sort_arr);
	
	return 0;
}
*/

