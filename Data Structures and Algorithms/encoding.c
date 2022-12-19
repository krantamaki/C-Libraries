#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "general.h"
#include "encoding.h"


// HAMMING CODE (UNDER CONSTRUCTION!)

// Encoder for bitwise data, which not only allows the detection of possible
// one- or two-bit errors but correction of single bit errors.

// Function for checking that the parity of the bit array is even
int _check_parity(void* ptr, size_t size) {
    int count = 0;
    for (size_t i = 0; i < size; i++) {
        char byte = *(char*)_apply(ptr, i, sizeof(char));
        for (int j = 0; j < 8; j++) {
            count += get_bit(byte, j);
        }
    }
    if (count % 2 == 0) return 1;
    else return 0;
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
    int n = size * sizeof(char);
    while (n + m > _pow(2, m) - 1) {
        m++;
    }

    // Then we must find the number of bytes for which memory is allocated.
    // That is bytes = ceil(n + m / 8) 
    int bytes = _ceil(n + m, 8);
    void* new_arr = (void*)malloc(bytes * sizeof(char));

    // Add the original data to the new array leaving places for the parity bits
    int org_i = 0;  // Goes from 0 to 8
    int new_i = 2;  // Goes from 0 to 8
    int len_i = 2;  // Covers the total length of the updated array
    int org_byte_i = 0;
    char org_byte = _apply(ptr, 0, sizeof(char));
    int new_byte_i = 0;
    char new_byte;
    while (org_byte_i < size) {
        // Check if new_i is a power of 2
        if (!(len_i & (len_i - 1))) {
            len_i++;
            new_i++;
        }
        else {
            int bit = get_bit(org_byte, org_i);
            if (bit) new_byte = toggle_bit(new_byte, new_i);
            new_i++;
            len_i++;
            org_i++;
        }
                    
        // Get new bytes if indeces exceed the size of a byte
        if (new_i == 8) {
			_swap(_apply(new_arr, new_byte_i, sizeof(char)), &new_byte, sizeof(char));
			new_byte = (char)0;
			new_byte_i++;
			new_i = 0;
		}
		if (org_i == 8) {
			org_byte_i++;
			org_byte = _apply(ptr, org_byte_i, sizeof(char));
			org_i = 0;
		}
    }
	return new_arr;
}

// Function removing the parity bits and returning the "original" bit array
// Allocates memory for the new array.
void* _reduce_from_parity(void* ptr, size_t size) {


}

// Single error correcting and double error detecting SECDED encoder
void* SECDED_encoder(void* org, size_t size) {

}

// Single error correcting and double error detecting SECDED decoder
void* SECDED_decoder(void* org, size_t size) {

}


