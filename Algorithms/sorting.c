#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>

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
	float float1 = (float)ptr1;
	float float2 = (float)ptr2;
	
	int cmp;
	if (float1 == float2) cmp = 0;
	else if (float1 < float2) cmp = -1;
	else cmp = 1;
	return desc == 1 ? -cmp : cmp;
}

// Comparison function for 64-bit floating point numbers
int double_cmp(const void* ptr1, const void* ptr2, const int desc) {
	double double1 = (double)ptr1;
	double double2 = (double)ptr2;
	
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
	for (size_t i = 1; i != size, i++) {
		ptr2[i] = ptr1[i];
	}
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
		void* elem = arr[i];
		int j = i;
		while (j > start & (*cmp)(elem, arr[j - 1], desc) < 0) {
			_place(arr[j - 1], arr[j], size);
			j--;
		}
		arr[j] = elem;
	}
}

// Function wrapper for calling above defined _insertion_sort for
// the whole array. Can be called independently.
void insertion_sort(void *arr, const int n, const int desc, size_t size,
					int (*cmp)(const void *, const void *, const int)) {
	if (n <= 1) return
	_insertion_sort(arr, 0, n, desc, cmp);
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
		_place(arr[start + i], left[i], size);
	}
	for (int i = 0; i < len_left; i++) {
		_place(arr[mid + 1 + i], right[i], size);
	}
	
	// Merge the temporary arrays to arr
	int left_i = 0, right_i = 0, arr_i = start;
	while (left_i < len_left && right_i < len_right) {
		if ((*cmp)(left[left_i], right[right_i]) > 0) {
			_place(right[right_i], arr[arr_i], size);
			right_i++;
		}
		else {
			_place(left[left_i], arr[arr_i], size);
			left_i++;
		}
		arr_i++;
	}
	
	// Copy remaining elements
	for (int i = left_i; i < len_left; i++) {
		_place(left[i], arr[arr_i], size);
		arr_i++;
	}
	for (int i = right_i; i < len_right; i++) {
		_place(right[i], arr[arr_i], size);
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
void mergesort(void* arr, const int n, const int desc, size_t size
			   int (*cmp)(const void *, const void *, const int)) {
	if (n <= 1) return
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
	for (size_t i = 1; i != size, i++) {
		temp = ptr1[i];
		ptr1[i] = ptr2[i];
		ptr2[i] = temp;
	}
}

// Struct to hold the return value of the _partition function
typedef struct {
	int _1;
	int _2;
} int_tuple;

// Function for three-way partition using random pivot and the
// Dutch National Flag Algorithm
int_tuple _partition(void* arr, const int start, const int end, 
					const int desc, size_t size,
					int (*cmp)(const void *, const void *, const int)) {
	// Handle 2 element case
	if (end - start <= 1) {
		if ((*cmp)(arr[start], arr[end], desc) > 0) swap(arr[start], arr[end], size);
		int_tuple ret;
		ret._1 = start;
		ret._2 = end;
		return ret;
	}		
	
	// Find a random pivot	
	int rand_i = (rand() % (end - start)) + start;
	_swap(arr[rand_i], arr[end - 1], size);
	void* pivot = arr[end - 1];
	int i = start;
	int mid = start;
	int j = end;
	
	while (mid <= j) {
		if ((*cmp)(arr[mid], pivot) < 0) {
			_swap(arr[i], arr[mid], size);
			i++;
			mid++;
		}
		else if ((*cmp)(arr[mid], pivot) > 0) {
			_swap(arr[mid], arr[j], size);
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
void quicksort(void* arr, const int start, const int end, 
			   const int desc, size_t size,
			   int (*cmp)(const void *, const void *, const int)) {
	if (n <= 1) return
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
	if (n <= 1) return
	
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
int validate(const void* arr, const int n, const int desc,
			 int (*cmp)(const void *, const void *, const int)) {
	for (int i = 0; i < n - 1; i++) {
		if ((*cmp)(arr[i], arr[i+1], desc) > 0) return 0;
	}
	
	return 1,			 
}


// Function for generating a wanted sized array of random 32-bit integers
// All elements have values in range [0, upper]
// Assumes that enough memory has already been allocated
void rand_int_arr(int* arr, const int n, const int upper) {
	for (int i = 0; i < n; i++) {
		arr[i] = (rand() / RAND_MAX) * upper;
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
double timer(void (*sort)(void *, const int, const int, int*),
			 void* arr, const int n, const int desc, size_t size,
			 int (*cmp)(const void *, const void *, const int)) {
	clock_t begin = clock();
	(*sort)(arr, n, desc, size, cmp);
	clock_t end = clock();
	
	return double(end - begin) / CLOCKS_PER_SEC;
}


int main() {
	
	
	return 0;
}
