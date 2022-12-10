#ifndef SEARCHING
#define SEARCHING

int _binary_search(void* arr, void* key, const int start, 
				   const int end, size_t size, int (*cmp)(void*, void*));
				   
int binary_search(void* arr, void* key, const int n, 
				  size_t size, int (*cmp)(void*, void*));
					   
int binary_search_low(void* arr, void* key, const int n, 
					  size_t size, int (*cmp)(void*, void*));
					    
int binary_search_high(void* arr, void* key, const int n, 
					   size_t size, int (*cmp)(void*, void*));
					   
int _partition_2way(void* arr, const int start, const int end, 
					size_t size, int (*cmp)(void*, void*));

void* _quickselect(void* arr, const int start, const int end, const int k, 
				   size_t size, int (*cmp)(void*, void*));
			
void* quickselect(void* arr, const int n, const int k, 
				  size_t size, int (*cmp)(void*, void*));
				  
int _peakfinder(void* arr, const int start, const int end, 
				size_t size, int (*cmp)(void*, void*));
				
int peakfinder(void* arr, const int n, size_t size, 
			   int (*cmp)(void*, void*));
			   
//int main();

#endif
