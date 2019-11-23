#include <vector_utils.hpp>

std::vector<double> create_cosine_vector(unsigned int size){
    std::vector<double> ret(size);
    for(unsigned int i = 0; i < size; i++){
        ret[i] = cos(i * i);
    }
    return ret;
}

std::vector<double> create_sin_vector(unsigned int size){
    std::vector<double> ret(size);
    for(unsigned int i = 0; i < size; i++){
        ret[i] = sin(i * i);
    }
    return ret;
}

double compute_sum(const std::vector<double> &v){
    double ret = 0.0;
    for(const auto &el : v){
        ret += el;
    }
    return ret;
}

double compute_L2_norm(const std::vector<double> &v){
    double ret = 0.0;
    for(const auto &el : v){
        ret += el * el;
    }
    return sqrt(ret);
}

double compute_L1_norm(const std::vector<double> &v){
    double ret = 0.0;
    for(const auto &el : v){
        ret += fabs(el);
    }
    return ret;
}

double compute_L_inf_norm(const std::vector<double> &v){
    double ret = 0.0;
    for(const auto &el : v){
        ret = ret > fabs(el) ? ret : fabs(el);
    }
    return ret;
}
