#include <ops_utils.hpp>

double dot(const std::vector<double> &v1, const std::vector<double> &v2){
    if(v1.size() != v2.size())
        throw std::logic_error("vectors must have equal sizes");

    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

std::vector<double> axpby(
    const double &a, const std::vector<double> &x, const double &b, const std::vector<double> &y){
    
    if(x.size() != y.size())
        throw std::logic_error("vectors must have equal sizes");
    std::vector<double> ret(x.size());
    axpby_store(ret, a, x, b, y);
    return ret;
}

void axpby_store(
    std::vector<double> &res,
    const double &a, const std::vector<double> &x, const double &b, const std::vector<double> &y){
    
    if(x.size() != y.size() || x.size() != res.size())
        throw std::logic_error("vectors must have equal sizes");
    
    for(unsigned int i = 0; i < x.size(); i++){
        res[i] = a * x[i] + b * y[i];
    }
}

std::vector<double> SpMV(const ELLMatrix &A, const std::vector<double> &x){
    // check commented due to mpi realisation
    // if(A.size != x.size())
    //     throw std::logic_error("vector and matrice must have equal sizes");
    std::vector<double> ret(x.size(), 0.0);
    SpMV_store(ret, A, x);
    return ret;
}

void SpMV_store(std::vector<double> &res, const ELLMatrix &A, const std::vector<double> &x){
    // checks commented due to mpi realisation
    // if(&res == &x)
    //     throw std::logic_error("result and input vector must be different objects");
    // if(A.size != x.size() || x.size() != res.size())
    //     throw std::logic_error("vectors and matrice must have equal sizes");
    
    for(unsigned int i = 0; i < A.size; i++){
        double sum = 0.0;

        for(unsigned int j = 0; j < A.nonnull_els_in_row; j++){
            const ColVal &c = A.get_elem_from_row(i, j);
            if(c.col < 0){
                continue;
            }
            sum += c.val * x[c.col];
        }
        res[i] = sum;
    }
}

std::vector<double> inv_diag_SpMV(const ELLMatrix &A, const std::vector<double> &x){
    if(A.size != x.size())
        throw std::logic_error("vector and matrice must have equal sizes");
    std::vector<double> ret(x.size());
    inv_diag_SpMV_store(ret, A, x);
    return ret;
}

void inv_diag_SpMV_store(std::vector<double> &res, const ELLMatrix &A, const std::vector<double> &x){
    // checks commented due to mpi realisation
    // if(A.size != x.size() || x.size() != res.size())
    //     throw std::logic_error("vectors and matrice must have equal sizes");
    
     for(unsigned int i = 0; i < A.size; i++){
        const ColVal &cv = A.get_diagonal_element(i);
        res[i] = 1 / cv.val * x[i];
    }
}
