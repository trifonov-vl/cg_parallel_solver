#pragma once
#include <vector_utils.h>
#include <ell_utils.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

double omp_dot(const struct Vector v1, const struct Vector v2);

struct Vector omp_axpby(double a, const struct Vector x, double b, const struct Vector y);

void omp_axpby_store(const struct Vector res, double a, const struct Vector x, double b, const struct Vector y);

struct Vector omp_SpMV(const struct ELLMatrix A, const struct Vector x);

void omp_SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x);

struct Vector omp_inv_diag_SpMV(const struct ELLMatrix A, const struct Vector x);

void omp_inv_diag_SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x);
