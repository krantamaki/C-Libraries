#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "general.h"
#include "algorithms.h"


// BINARY SEARCH And VARIATIONS

// Recursively called helper function for classic binary search
int _binary_search(void* arr, void* key, const int start, 
				   const int end, size_t size, int (*cmp)(void*, void*)) {
	const int mid = (end + start) / 2;
	if (start <= end) {
		// Compare the value to be found to the new mid point
		int cmp_val = (*cmp)(_apply(arr, mid, size), key, size);
		
		// If values are equal return true
		if (cmp_val == 0) return 1;
		
		// Else if arr[mid] < key search the right subarray
		else if (cmp_val < 0) _binary_search(arr, key, mid + 1, end, size, cmp);
		
		// Otherwise search left subarray
		else _binary_search(arr, key, start, mid - 1, size, cmp);
	}
	else return 0;
}

// Main binary search function. Returns 1 if key in array, 0 otherwise.
// Works in O(log_2(n)) time
int binary_search(void* arr, void* key, const int n, 
				  size_t size, int (*cmp)(void*, void*)) {
	return _binary_search(arr, key, 0, n, size, cmp);
}


// Recursively called helper function for binary_search_low
int _binary_search_low(void* arr, void* key, const int start, const int end, 
					   int low, size_t size, int (*cmp)(void*, void*)) {
	const int mid = (end + start) / 2;
	if (start <= end) {
		// Compare the value to be found to the new mid point
		int cmp_val = (*cmp)(_apply(arr, mid, size), key, size);
		
		// If the key is found return it
		if (cmp_val == 0) return mid;
		
		// Else if arr[mid] < key arr[mid] is the new lowest found, but continue
		// searching the left subarray
		else if (cmp_val < 0) _binary_search_low(arr, key, start, mid - 1, mid, size, cmp);
		
		// Otherwise no new low was found, but continue looking through the
		// right subarray
		else _binary_search_low(arr, key, mid + 1, end, low, size, cmp);
	}
	else low;				   
}

// Modified binary search that finds the greatest element s.t. elem <= key
// from the array. Returns the index of the element or -1 if all elements in
// arr are greater than key. Works in O(log_2(n)) time.
int binary_search_low(void* arr, void* key, const int n, 
					  size_t size, int (*cmp)(void*, void*)) {
	return _binary_search_low(arr, key, 0, n, -1, size, cmp);
}


// Recursively called helper function for binary_search_high
int _binary_search_high(void* arr, void* key, const int start, const int end, 
					    int high, size_t size, int (*cmp)(void*, void*)) {
	const int mid = (end + start) / 2;
	if (start <= end) {
		// Compare the value to be found to the new mid point
		int cmp_val = (*cmp)(_apply(arr, mid, size), key, size);
		
		// If the key is found return it
		if (cmp_val == 0) return mid;
		
		// Else if arr[mid] > key arr[mid] is the new highest found, but continue
		// searching the right subarray
		else if (cmp_val > 0) _binary_search_low(arr, key, mid + 1, end, mid, size, cmp);
		
		// Otherwise no new low was found, but continue looking through the
		// left subarray
		else _binary_search_low(arr, key, start, mid - 1, high, size, cmp);
	}
	else high;				   
}

// Modified binary search that finds the smallest element s.t. elem >= key
// from the array. Returns the index of the element or -1 if all elements in
// arr are smaller than key. Works in O(log_2(n)) time.
int binary_search_high(void* arr, void* key, const int n, 
					   size_t size, int (*cmp)(void*, void*)) {
	return _binary_search_high(arr, key, 0, n, -1, size, cmp);
}


// QUICKSELECT

// Standard two-way partition. The three-way partition from sorting.c
// would also work, but then this file would have dependency on sorting.h
// and to avoid that new algorithm is implemented here
void _partition_2way()


