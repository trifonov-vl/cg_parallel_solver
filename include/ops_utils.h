#include <vector_utils.h>
#include <ell_utils.h>
#include <stdlib.h>

double dot(const struct Vector v1, const struct Vector v2);

struct Vector axpby(double a, const struct Vector x, double b, const struct Vector y);

struct Vector SpMV(const struct ELLMatrix A, const struct Vector x);
