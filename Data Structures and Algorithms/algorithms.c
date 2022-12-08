#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "general.h"
#include "algorithms.h"


// SEARCH FUNCTIONS

int _binary_search(void* arr, const int start, const int end, 
				   size_t size, int (*cmp)(void*, void*)) {
	const it mid = (end + start) / 2;
				   
}

