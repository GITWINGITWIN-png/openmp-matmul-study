# Build the OpenMP matrix-multiplication program.
CC     ?= gcc
OPT    ?= -O3 -march=native
CFLAGS  = $(OPT) -fopenmp -Wall -Wextra -std=c11
LDLIBS  = -lm

parallel: parallel.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f parallel

.PHONY: clean
