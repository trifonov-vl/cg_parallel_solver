#pragma once
#include <ell_utils.hpp>
#include <ops_utils.hpp>
#include <vector>
#include <stdexcept>

int solve(
    const ELLMatrix &A, const std::vector<double> &b, 
    std::vector<double> &x, const double &eps, const int &max_it);