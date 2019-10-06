#pragma once
#include <vector_utils.h>
#include <ell_utils.h>
#include <stdlib.h>
#include <assert.h>

double dot(const struct Vector v1, const struct Vector v2);

struct Vector axpby(double a, const struct Vector x, double b, const struct Vector y);

void axpby_store(const struct Vector res, double a, const struct Vector x, double b, const struct Vector y);

struct Vector SpMV(const struct ELLMatrix A, const struct Vector x);

void SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x);

struct Vector inv_diag_SpMV(const struct ELLMatrix A, const struct Vector x);

void inv_diag_SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x);
