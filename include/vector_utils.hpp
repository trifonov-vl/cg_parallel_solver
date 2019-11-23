#pragma once
#include <cmath>
#include <vector>

std::vector<double> create_cosine_vector(const unsigned int &size);

std::vector<double> create_sin_vector(const unsigned int &size);

double compute_sum(const std::vector<double> &v);

double compute_L2_norm(const std::vector<double> &v);

double compute_L1_norm(const std::vector<double> &v);

double compute_L_inf_norm(const std::vector<double> &v);
