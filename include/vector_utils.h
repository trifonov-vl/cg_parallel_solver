#pragma once
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct Vector {
    double *vals;
    unsigned int size;
};

struct Vector create_uninit_Vector(unsigned int size);

struct Vector create_const_Vector(double val, unsigned int size);

struct Vector create_cosine_Vector(unsigned int size);

struct Vector create_sin_Vector(unsigned int size);

struct Vector copy_Vector(const struct Vector v);

void copy_from_Vector_to_Vector(const struct Vector from, const struct Vector to);

void delete_Vector(struct Vector *v);

double compute_sum(const struct Vector v);

double compute_L2_norm(const struct Vector v);

double compute_L1_norm(const struct Vector v);

double compute_L_inf_norm(const struct Vector v);
