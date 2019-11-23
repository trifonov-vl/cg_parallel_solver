#pragma once
#include <ell_utils.hpp>
#include <numeric>
#include <vector>
#include <stdexcept>

double dot(const std::vector<double> &v1, const std::vector<double> &v2);

std::vector<double> axpby(
    const double &a, const std::vector<double> &x, const double &b, const std::vector<double> &y);

void axpby_store(
    std::vector<double> &res,
    const double &a, const std::vector<double> &x, const double &b, const std::vector<double> &y);

std::vector<double> SpMV(const ELLMatrix &A, const std::vector<double> &x);

void SpMV_store(std::vector<double> &res, const ELLMatrix &A, const std::vector<double> &x);

std::vector<double> inv_diag_SpMV(const ELLMatrix &A, const std::vector<double> &x);

void inv_diag_SpMV_store(std::vector<double> &res, const ELLMatrix &A, const std::vector<double> &x);
