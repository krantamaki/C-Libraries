CC      = gcc                # Choose the compiler
CFLAGS  = -fopenmp -Wall     # Needed C-flags like openMP
LIBS    = -lm                # Needed libraries
SRCS    = $(wildcard *.c)    # Collect all .c files from current directory
GEN     = '../../general.c'  # Path to the general.c file with useful functions

matrix.o:
	$(CC) $(CFLAGS) $(SRCS) $(GEN) $(LIBS) -o $@
