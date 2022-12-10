#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "general.h"
#include "searching.h"
// For testing purposes include the sorting library.
// #include "sorting.h"


// BINARY SEARCH AND IT'S VARIATIONS

// Recursively called helper function for classic binary search
int _binary_search(void* arr, void* key, const int start, 
				   const int end, size_t size, int (*cmp)(void*, void*)) {
	const int mid = (end + start) / 2;
	if (start <= end) {
		// Compare the value to be found to the new mid point
		int cmp_val = (*cmp)(_apply(arr, mid, size), key);
		
		// If values are equal return true
		if (cmp_val == 0) return mid;
		
		// Else if arr[mid] < key search the right subarray
		else if (cmp_val < 0) return _binary_search(arr, key, mid + 1, end, size, cmp);
		
		// Otherwise search left subarray
		else return _binary_search(arr, key, start, mid - 1, size, cmp);
	}
	else return -1;
}

// Main binary search function. Returns the index of the key if found from
// the array, -1 otherwise.
// Works in O(log_2(n)) time
int binary_search(void* arr, void* key, const int n, 
				  size_t size, int (*cmp)(void*, void*)) {
	return _binary_search(arr, key, 0, n - 1, size, cmp);
}

// Modified iterative binary search that finds the greatest element 
// s.t. elem <= key from the array. Returns the index of the element 
// or -1 if all elements in arr are greater than the key. 
// Works in O(log_2(n)) time.
int binary_search_low(void* arr, void* key, const int n, 
					  size_t size, int (*cmp)(void*, void*)) {
	int ret = -1;
	int start = 0;
	int end = n - 1;
	while (start <= end) {
		int mid = (start + end) / 2;
		int cmp_val = (*cmp)(key, _apply(arr, mid, size));
		if (cmp_val == 0) {
			ret = mid;
			break;
		}
		else if (cmp_val > 0) {
			ret = mid;
			start = mid + 1;
		}
		else end = mid - 1;
	}
	
	return ret;
}

// Modified iterative binary search that finds the smallest element 
// s.t. elem >= key from the array. Returns the index of the element 
// or -1 if all elements in arr are smaller than the key. 
// Works in O(log_2(n)) time.
int binary_search_high(void* arr, void* key, const int n, 
					   size_t size, int (*cmp)(void*, void*)) {
	int ret = -1;
	int start = 0;
	int end = n - 1;
	while (start <= end) {
		int mid = (start + end) / 2;
		int cmp_val = (*cmp)(key, _apply(arr, mid, size));
		if (cmp_val == 0) {
			ret = mid;
			break;
		}
		else if (cmp_val < 0) {
			ret = mid;
			end = mid - 1;
		}
		else start = mid + 1;
	}
	
	return ret;
}


// QUICKSELECT

// Standard two-way partition. The three-way partition from sorting.c
// would also work, but then this file would have dependency on sorting.h
// and to avoid that new algorithm is implemented here
int _partition_2way(void* arr, const int start, const int end, 
					size_t size, int (*cmp)(void*, void*)) {
	// Find a random pivot
	int rand_i = (rand() % (end - start)) + start;
	_swap(_apply(arr, rand_i, size), _apply(arr, end, size), size);
	void* pivot = _apply(arr, end, size);
	int i = start;
	
	for (int j = start; j < end; j++) {
		if ((*cmp)(_apply(arr, j, size), pivot) <= 0) {
			_swap(_apply(arr, i, size), _apply(arr, j, size), size);
			i++;
		}
	}
	_swap(_apply(arr, i, size), _apply(arr, end, size), size);
	return i;			
}

// Recursively called helper function for quickselect
void* _quickselect(void* arr, const int start, const int end, const int k, 
				   size_t size, int (*cmp)(void*, void*)) {
	// If there is only one element return it
	if (start >= end) return _apply(arr, start, size);
	
	int pivot_i = _partition_2way(arr, start, end, size, cmp);
	
	// If the pivot index is k return pointer to said element
	if (k == pivot_i) return _apply(arr, pivot_i, size);
	
	// Otherwise search through the correct subarray
	else if (k <= pivot_i) return _quickselect(arr, start, pivot_i - 1, k, size, cmp);
	
	else return _quickselect(arr, pivot_i + 1, end, k, size, cmp);
	
	
					  
}

// Standard quickselect algorithm. Finds the k:th smallest element in an unsorted.
// array and places it at the correct index. Returns the pointer to this element
// if k < n and null pointer otherwise.
// Note! Rest of the elements are not sorted. 
// Has average time complexity of O(n).
void* quickselect(void* arr, const int n, const int k, 
				  size_t size, int (*cmp)(void*, void*)) {
	if (k >= n) return NULL;
	
	return _quickselect(arr, 0, n - 1, k, size, cmp);
}


// PEAKFINDER

// Recursively called helper function for peakfinder
int _peakfinder(void* arr, const int start, const int end, 
				size_t size, int (*cmp)(void*, void*)) {
	const int mid = (end + start) / 2;
	int cmp_p = (*cmp)(_apply(arr, mid, size), _apply(arr, mid + 1, size));
	int cmp_n = (*cmp)(_apply(arr, mid, size), _apply(arr, mid - 1, size));
	
	// If the middle element is the peak return it's index
	if (cmp_n >= 0 && cmp_p >= 0) {
		return mid;
	}
	
	// If the right neighbour is larger than middle then the right sub-array
	// must contain the peak
	else if (cmp_p < 0) return _peakfinder(arr, mid + 1, end, size, cmp);
	
	// Else the left neighbour is larger than middle and thus the left 
	// sub-array must contain the peak
	else return _peakfinder(arr, start, mid - 1, size, cmp);
}

// Algorithm for finding the peak of an array ie. the element that is 
// greater or equal to both of it's neighbours. Returns the index of
// the peak or -1 if n <= 1
// Works in O(log_2(n))
int peakfinder(void* arr, const int n, size_t size, 
			   int (*cmp)(void*, void*)) {
	// Check that there is at least 2 elements in the array
	if (n < 2) return -1;
				   
	// Check the border cases for mid = 0 and mid = n, where there is 
	// only one neighbour
	if ((*cmp)(_apply(arr, 0, size), _apply(arr, 1, size)) >= 0) {
		return 0;
	}
	else if ((*cmp)(_apply(arr, n - 2, size), _apply(arr, n - 1, size)) < 0) {
		return n - 1;
	}
	// Otherwise recursively go through the array
	else {
		return _peakfinder(arr, 1, n - 2, size, cmp);
	}
}


/*
// MAIN FUNCTION (ONLY FOR TESTING PURPOSES)
// To compile this: gcc -fopenmp -Wall searching.c sorting.c general.c -o searching.o
int main() {
	time_t t = time(NULL);
	srand((unsigned) t);
	int n = 100;
	int m = 250;
	// Create an array of n random integers on the interval [0, m]
	int* arr = (int*)malloc(n * sizeof(int));
	rand_int_arr(arr, n, m);
	
	// Copy said array and sort it
	int* sorted_arr = (int*)malloc(n * sizeof(int));
	memcpy(sorted_arr, arr, n * sizeof(int));
	insertion_sort(sorted_arr, n, sizeof(int), int_cmp);
	
	// We will use binary search to try and find the number k
	// so linearly go through the sorted array and find the closest 
	// values from above and below to it in the array
	int k = 50;
	int exact = -1;
	int lower = -1;
	int upper = -1;
	for (int j = 0; j < n; j++) {
		if (sorted_arr[j] == k) {
			exact = j;
			lower = j;
			upper = j;
			break;
		}
		else if (sorted_arr[j] < k) {
			lower = j;
		}
		else {
			upper = j;
			break;
		}
	}
	
	// TEST BINARY SEARCH ALGORITHMS
	printf("\nTesting the binary search algorithms\n");
	
	// Test classic binary search
	int b_exact = binary_search(sorted_arr, &k, n, sizeof(int), int_cmp);
	printf("Classic binary search: ");
	(b_exact == exact) ? printf("Valid\n") : printf("Invalid\n");
	
	// Test lower bound search
	int b_lower = binary_search_low(sorted_arr, &k, n, sizeof(int), int_cmp);
	printf("Binary search for lower bound: ");
	(b_lower == lower) ? printf("Valid\n") : printf("Invalid (%d != %d)\n", b_lower, lower);
	
	// Test upper bound search
	int b_upper = binary_search_high(sorted_arr, &k, n, sizeof(int), int_cmp);
	printf("Binary search for upper bound: ");
	(b_upper == upper) ? printf("Valid\n") : printf("Invalid (%d != %d)\n", b_upper, upper);
	
	// TEST QUICKSELECT
	printf("\nTesting the quickselect algorithm: ");
	int i = 50;
	int elem_i = sorted_arr[i];
	int q_elem = *(int*)quickselect(arr, n, i, sizeof(int), int_cmp);
	(elem_i == q_elem) ? printf("Valid\n") : printf("Invalid (%d != %d)\n", elem_i, q_elem);
	
	// TEST PEAKFINDER
	printf("\nTesting the peakfinder algorithm: ");
	int p_peak = peakfinder(arr, n, sizeof(int), int_cmp);
	
	// Check that this is indeed a peak
	int is_peak;
	if (p_peak == 0) is_peak = (arr[p_peak] >= arr[p_peak + 1]) ? 1 : 0;
	else if (p_peak == n - 1) is_peak = (arr[p_peak] >= arr[p_peak - 1]) ? 1 : 0;
	else is_peak = (arr[p_peak] >= arr[p_peak - 1] && arr[p_peak] >= arr[p_peak + 1]) ? 1 : 0;
	
	(is_peak) ? printf("Valid\n") : printf("Invalid\n");
	
	return 0;
}
*/
