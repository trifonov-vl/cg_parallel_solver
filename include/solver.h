#pragma once
#include <ell_utils.h>
#include <ops_utils.h>
#include <vector_utils.h>
#include <assert.h>

int solve(const struct ELLMatrix A, const struct Vector b, const struct Vector x, double eps, int max_it);