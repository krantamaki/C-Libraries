#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "general.h"
#include "encoding.h"


// HAMMING CODE

// Encoder for bitwise data, which not only allows the detection of possible
// one- or two-bit errors but correction of single bit errors.

// Function for checking that the parity of the bit array is even
int _check_parity(void* ptr, size_t size) {
    int count = 0;
    for (size_t i = 0; i < size; i++) {
        char byte = *(char*)_apply(ptr, i, sizeof(char));
        for (int j = 0; j < 8; j++) {
            count += (byte >> j) & 1;
        }
    }
    if (count % 2 == 0) return 1;
    else return 0;
} 

// Function for computing the power for a given integer
int _pow(int x, int n) {
    int ret = 1;
    for (int i = 0; i < n; i++) {
        ret *= x;
    }

    return ret;
}

// Function for "dividing up"
int _ceil(int a, int b) {
    return (a + b - 1) / b;
}

// Function for extending the bit array to needed size with places for parity
// bits added. The parity bits are initialized to zeros. 
// Allocates memory for the new array.
void* _extend_to_parity(void* ptr, size_t size) {
    // Parity bits are added as powers of two of the number of data bits.
    // That is bits 1, 2, 4, 8, 16 ... are parity bits. If the number of 
    // data bits is n then the new size is n + m where m is solved from 
    // n = 2^m - m - 1. As this is non-trivial to solve we will just loop
    // over possible m values until one satisfying the demands is found
    int m = 1;
    int n = size * 8;
    while (n + m < _pow(2, m) - 1) {
        m++;
    }

    // Then we must find the number of bytes for which memory is allocated.
    // That is bytes = ceil(n + m / 8) 
    int bytes = _ceil(n + m, 8);
    void* new_arr = (void*)malloc(bytes * sizeof(char));

    // Add the original data to the new array leaving places for the parity bits
    int org_i = 0;
    int new_i = 2;
    while (org_i < n) {
        // Check if new_i is a power of 2
        if (!(new_i & (new_i - 1))) {
            new_i++;
        }
        else {
            
        }
    }

}

// Function removing the parity bits and returning the "original" bit array
// Allocates memory for the new array.
void* _reduce_from_parity(void* ptr, size_t size) {


}

// Single error correcting and double error detecting SECDED encoder
void* SECDED_encoder(void* org, size_t size) {

}


