#pragma once
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

struct ColVal{
    int col;
    double val;
};

struct ELLMatrix{
    unsigned int size;
    unsigned int nonnull_els_in_row;
    struct ColVal *colvals;
    unsigned int *diag_indices;
};

void delete_ELL(struct ELLMatrix *m);

void print_ELL(const struct ELLMatrix m);

struct ELLMatrix generate_ELL_3D_DECART(unsigned int Nx, unsigned int Ny, unsigned int Nz);
