#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "sorting.h"

// The comparison functions for datatypes int, float, double and char*.
// Comparison functions are passed to the sorting algorithms and must
// correspond with the datatype of the elements of the array

// Comparison function for 32-bit signed integers
int int_cmp(const void* ptr1, const void* ptr2, const int desc) {
	int int1 = (int)ptr1;
	int int2 = (int)ptr2;
	return desc == 1 ? int2 - int1 : int1 - int2;
}

// Comparison function for 32-bit floating point numbers
int float_cmp(const void* ptr1, const void* ptr2, const int desc) {
	float float1 = *(const float*)ptr1;
	float float2 = *(const float*)ptr2;
	
	int cmp;
	if (float1 == float2) cmp = 0;
	else if (float1 < float2) cmp = -1;
	else cmp = 1;
	return desc == 1 ? -cmp : cmp;
}

// Comparison function for 64-bit floating point numbers
int double_cmp(const void* ptr1, const void* ptr2, const int desc) {
	double double1 = *(const double*)ptr1;
	double double2 = *(const double*)ptr2;
	
	int cmp;
	if (double1 == double2) cmp = 0;
	else if (double1 < double2) cmp = -1;
	else cmp = 1;
	return desc == 1 ? -cmp : cmp;
}

// Comparison function for NULL TERMINATED char arrays
// Differs from string.h libraries strcmp() function by comparing
// the sum of the byte representations of the chars rather that finding 
// the first different chars and comparing them
int str_cmp(const void* ptr1, const void* ptr2, const int desc) {
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

	return int_cmp((void*)sum1, (void*)sum2, desc);
}


// Necessary functions for insertion sort

// Function for placing the value of src into the pointer of dst
void _place(void* src, void* dst, size_t size) {
	unsigned char *ptr1 = src, *ptr2 = dst;
	for (size_t i = 1; i != size; i++) {
		ptr2[i] = ptr1[i];
	}
}

// Function for getting the ith element of a void pointer array
void* _apply(void* arr, const int i, size_t size) {
	return (void*)((char*)arr + (int)size * i);
}

// Insertion sort algorithm for sorting subarray arr[start, end]. 
// This is defined first as it is what quick- and mergesort algorithms 
// will default to with small enough array sizes. 
// Should not be called independently
// Space complexity of O(1)
// Best case time complexity O(n) and worst case O(n^2) so not 
// recommended for larger arrays. NOTE! Not parallelized.
void _insertion_sort(void *arr, const int start, const int end, 
					 const int desc, size_t size, 
					 int (*cmp)(const void *, const void *, const int)) {
	for (int i = start; i < end; i++) {
		void* elem = _apply(arr, i, size);
		int j = i;
		while (j > start & (*cmp)(elem, _apply(arr, j - 1, size), desc) < 0) {
			_place(_apply(arr, j - 1, size), _apply(arr, j, size), size);
			j--;
		}
		_place(elem, _apply(arr, j, size), size);
	}
}

// Function wrapper for calling above defined _insertion_sort for
// the whole array. Can be called independently.
void insertion_sort(void *arr, const int n, const int desc, size_t size,
					int (*cmp)(const void *, const void *, const int)) {
	if (n <= 1) return;
	_insertion_sort(arr, 0, n, desc, size, cmp);
}


// Necessary functions for mergesort. The functions with name starting
// with '_' should not be called independently, but are subroutines of 
// of the main mergesort() function

// Function for merging two sub arrays (arr[start:mid + 1] and arr[mid + 1: end])
// in the correct order
void _merge(void* arr, const int start, const int mid, const int end,
			const int desc, size_t size, 
			int (*cmp)(const void *, const void *, const int)) {
	int len_left = mid - start;
	int len_right = end - 1 - mid;
	// Allocate temporary arrays
	void* left = (void*)malloc((len_left) * size);
	void* right = (void*)malloc((len_right) * size);
	
	// Copy the data to the temp arrays
	for (int i = 0; i < len_left; i++) {
		_place(_apply(arr, start + i, size), _apply(left, i, size), size);
	}
	for (int i = 0; i < len_left; i++) {
		_place(_apply(arr, mid + 1 + i, size), _apply(right, i, size), size);
	}
	
	// Merge the temporary arrays to arr
	int left_i = 0, right_i = 0, arr_i = start;
	while (left_i < len_left && right_i < len_right) {
		if ((*cmp)(_apply(left, left_i, size), _apply(right, right_i, size), size) > 0) {
			_place(_apply(right, right_i, size), _apply(arr, arr_i, size), size);
			right_i++;
		}
		else {
			_place(_apply(left, left_i, size), _apply(arr, arr_i, size), size);
			left_i++;
		}
		arr_i++;
	}
	
	// Copy remaining elements
	for (int i = left_i; i < len_left; i++) {
		_place(_apply(left, i, size), _apply(arr, arr_i, size), size);
		arr_i++;
	}
	for (int i = right_i; i < len_right; i++) {
		_place(_apply(right, i, size), _apply(arr, arr_i, size), size);
		arr_i++;
	}
	
	// Free allocated memory
	free(left);
	free(right);
}

// Recursively called assisting function for mergesort
void _mergesort(void* arr, const int start, const int end, 
				const int desc, size_t size,
				int (*cmp)(const void *, const void *, const int)) {
	if (start < end) {  // Sanity check (should always be the case)
		if (end - start > THRESHOLD) {  // TODO: Define THRESHOLD in header
			int mid = (end - start) / 2;
			#pragma omp taskgroup  // Start multiple recursive tasks in parallel
			{
				#pragma omp task shared(arr)  // Tasks share the same array
				_mergesort(arr, start, mid + 1, desc, size, cmp);
				#pragma omp task shared(arr)
				_mergesort(arr, mid + 1, end, desc, size, cmp);
			}
			// Merge the sorted subarrays
			_merge(arr, start, mid, end, desc, size, cmp);
		}
		// If the arrays are small enough just use insertion sort
		else _insertion_sort(arr, start, end, desc, size, cmp);
	}
}

// The main mergesort function
// As this is not an in-place implementation the space complexity is O(n)
// Time complexity is O(n*log_2(n)). Parallelized
void mergesort(void* arr, const int n, const int desc, size_t size,
			   int (*cmp)(const void *, const void *, const int)) {
	if (n <= 1) return;
	// Initialize parallelizations
	#pragma omp parallel
	#pragma omp single
	_mergesort(arr, 0, n, desc, size, cmp);
}


// Necessary functions for quick sort

// Function for swapping the references of two pointers
// Swaps the values byte by byte
// From: https://stackoverflow.com/questions/29596151/swap-function-using-void-pointers
void _swap(void* elem1, void* elem2, size_t size) {
	unsigned char *ptr1 = elem1, *ptr2 = elem2, temp;
	for (size_t i = 1; i != size; i++) {
		temp = ptr1[i];
		ptr1[i] = ptr2[i];
		ptr2[i] = temp;
	}
}


// Function for three-way partition using random pivot and the
// Dutch National Flag Algorithm
int_tuple _partition(void* arr, const int start, const int end, 
					const int desc, size_t size,
					int (*cmp)(const void *, const void *, const int)) {
	// Handle 2 element case
	if (end - start <= 1) {
		if ((*cmp)(_apply(arr, start, size), _apply(arr, end, size), desc) > 0) {
			_swap(_apply(arr, start, size), _apply(arr, end, size), size);
		}
		int_tuple ret;
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
		if ((*cmp)(_apply(arr, mid, size), pivot, desc) < 0) {
			_swap(_apply(arr, i, size), _apply(arr, mid, size), size);
			i++;
			mid++;
		}
		else if ((*cmp)(_apply(arr, mid, size), pivot, desc) > 0) {
			_swap(_apply(arr, mid, size), _apply(arr, j, size), size);
			j--;
		}
		else mid++;
	}
	
	int_tuple ret;
	ret._1 = i - 1;
	ret._2 = j + 1;
	
	return ret;
}

// Recursively called helper function for quicksort
void _quicksort(void* arr, const int start, const int end, 
				const int desc, size_t size,
				int (*cmp)(const void *, const void *, const int)) {
	if (start < end) {  // Sanity check
		if (end - start > THRESHOLD) {  // TODO: Define THRESHOLD in header
			int_tuple pivots = _partition(arr, start, end, desc, size, cmp);
			#pragma omp taskgroup  // Start multiple recursive tasks in parallel
			{
				#pragma omp task shared(arr)  // Tasks share the same array
				_quicksort(arr, start, pivots._1 + 1, desc, size, cmp);
				#pragma omp task shared(arr)
				_quicksort(arr, pivots._2, end, desc, size, cmp);
			}
		}
		// If the arrays are small enough just use insertion sort
		else _insertion_sort(arr, start, end, desc, size, cmp);
	}			
}

// The main quicksort function
// Space complexity O(1) and time complexity O(n*log_2(n)). Parallelized
void quicksort(void* arr, const int n, const int desc, size_t size,
			   int (*cmp)(const void *, const void *, const int)) {
	if (n <= 1) return;
	// Initialize parallelizations
	#pragma omp parallel
	#pragma omp single
	_quicksort(arr, 0, n, desc, size, cmp);	   		   
}


// Necessary functions for radix sorts

// Function for 32-bit signed integer lsd radix sort
// Space complexity O(n) and time complexity O(n)
// (as the number of possible elements 256 is considered insignificant)
// NOTE! Not parallelized (yet?)
// NOTE! The comparison function is not used but is included as parameter
// for compatibility
void int_radix_sort(void* arr, const int n, const int desc, size_t size,
			   int (*cmp)(const void *, const void *, const int)) {
	if (n <= 1) return;
	
	// Allocate memory for auxiliary arrays
	int* result = (int*)malloc(n * sizeof(int));
	int* temp = (int*)malloc(n * sizeof(int));  // Not actually necessary
	
	// Copy original array to temp array
	memcpy(temp, arr, sizeof(int));
	
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
		for (int i = 0; i < 256; i++) {
			start[i] = start[i - 1] + count[i - 1];
		}
		
		// Alter the output array
		for (int i = 0; i < n; i++) {
			num = temp[i];
			index = (num >> (8 * byte)) & 0xff;
			result[start[index]] = num;
			start[index]++;
		}
		
		// Swap references
		int* t = temp;
		temp = result;
		result = t;
	}
	
	// Toggle the sign bit back
	for (int i = 0; i < n; i++) temp[i] ^= 1 << 31;
	
	// Copy the result to original array
	memcpy(arr, temp, sizeof(int));
	
	// Free allocated memory
	free(temp);
	free(result);
	free(count);
	free(start);   
}

// TODO: Implement floating point radix sort


// Helper functions for the main function

// Function that validates that an array is indeed sorted
int validate_sort(const void* arr, const int n, const int desc, size_t size,
				  int (*cmp)(const void *, const void *, const int)) {
	for (int i = 0; i < n - 1; i++) {
		if ((*cmp)(_apply(arr, i, size), _apply(arr, i, size), desc) > 0) return 0;
	}
	
	return 1;			 
}


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


// Function for timing the execution time of a function 'sort'
// As this is used to time sorting algorithms the parameters needed
// to pass to the function 'sort' are also parameters here.
double sort_timer(void (*sort)(void *, const int, const int, size_t, int*),
				  void* arr, const int n, const int desc, size_t size,
				  int (*cmp)(const void *, const void *, const int)) {
	clock_t begin = clock();
	(*sort)(arr, n, desc, size, cmp);
	clock_t end = clock();
	
	return (double)(end - begin) / CLOCKS_PER_SEC;
}


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
	memcpy(small_sort_arr, small_arr, sizeof(int));
	
	// Insertion sort
	printf("\nSorted using insertion sort:\n");
	insertion_sort(small_sort_arr, 10, 0, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, sizeof(int));
	
	// Mergesort
	printf("\nSorted using mergesort:\n");
	mergesort(small_sort_arr, 10, 0, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, sizeof(int));
	
	// Quicksort
	printf("\nSorted using quicksort:\n");
	quicksort(small_sort_arr, 10, 0, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, sizeof(int));
	
	// Radix sort
	printf("\nSorted using radix sort:\n");
	int_radix_sort(small_sort_arr, 10, 0, sizeof(int), int_cmp);
	for (int i = 0; i < 10; i++) {
		printf("%d ", small_sort_arr[i]);
	}
	memcpy(small_sort_arr, small_arr, sizeof(int));
	
	// Test each algorithm using random array of 100 000 integers
	// and time them
	printf("\n\nTesting all algorithms with an array of 100 000 ints:\n");
	int* big_arr = (int*)malloc(100000 * sizeof(int));
	rand_int_arr(big_arr, 100000, 10);
	int* big_sort_arr = (int*)malloc(100000 * sizeof(int));
	memcpy(big_sort_arr, big_arr, sizeof(int));
	
	// Insertion sort
	printf("\nTesting insertion sort:\n");
	double exc_time = sort_timer(insertion_sort, big_sort_arr, 100000, 0, 
								 sizeof(int), int_cmp);
	int valid = validate_sort(big_sort_arr, 100000, 0, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	memcpy(big_sort_arr, big_arr, sizeof(int));
	
	// Merge sort
	printf("\nTesting mergesort:\n");
	exc_time = sort_timer(mergesort, big_sort_arr, 100000, 0, 
								 sizeof(int), int_cmp);
	valid = validate_sort(big_sort_arr, 100000, 0, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	memcpy(big_sort_arr, big_arr, sizeof(int));
	
	// Quicksort
	printf("\nTesting quicksort:\n");
	exc_time = sort_timer(quicksort, big_sort_arr, 100000, 0, 
								 sizeof(int), int_cmp);
	valid = validate_sort(big_sort_arr, 100000, 0, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	memcpy(big_sort_arr, big_arr, sizeof(int));
	
	// Radix sort
	printf("\nTesting radix sort:\n");
	exc_time = sort_timer(int_radix_sort, big_sort_arr, 100000, 0, 
								 sizeof(int), int_cmp);
	valid = validate_sort(big_sort_arr, 100000, 0, sizeof(int), int_cmp);
	if (valid) {
		printf("Valid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	else {
		printf("Invalid. Execution time: %.2f ms\n", exc_time * 100000.0);
	}
	memcpy(big_sort_arr, big_arr, sizeof(int));
	
	// Free allocated memory
	free(small_arr);
	free(small_sort_arr),
	free(big_arr);
	free(big_sort_arr);
	
	return 0;
}