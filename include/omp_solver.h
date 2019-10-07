#pragma once
#include <ell_utils.h>
#include <omp_ops_utils.h>
#include <vector_utils.h>
#include <assert.h>

int omp_solve(const struct ELLMatrix A, const struct Vector b, const struct Vector x, double eps, int max_it);