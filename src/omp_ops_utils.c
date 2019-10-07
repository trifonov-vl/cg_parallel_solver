#include <omp_ops_utils.h>

double omp_dot(const struct Vector v1, const struct Vector v2){
    assert(v1.size == v2.size);
    double res = 0.0;

    #pragma omp parallel for reduction(+:res)
    for(unsigned int i = 0; i < v1.size; i++){
        res += v1.vals[i] * v2.vals[i];
    }

    return res;
}

struct Vector omp_axpby(double a, const struct Vector x, double b, const struct Vector y){
    assert(x.size == y.size);
    struct Vector ret = create_uninit_Vector(x.size);
    omp_axpby_store(ret, a, x, b, y);
    return ret;
}

void omp_axpby_store(const struct Vector res, double a, const struct Vector x, double b, const struct Vector y){
    assert(x.size == y.size && y.size == res.size);
    
    #pragma omp parallel for
    for(unsigned int i = 0; i < x.size; i++){
        res.vals[i] = a * x.vals[i] + b * y.vals[i];
    }
}

struct Vector omp_SpMV(const struct ELLMatrix A, const struct Vector x){
    assert(A.size == x.size);
    struct Vector ret = create_uninit_Vector(x.size);
    omp_SpMV_store(ret, A, x);
    return ret;
}

void omp_SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x){
    assert(res.vals != x.vals);
    assert(res.size == x.size && x.size == A.size);

    #pragma omp parallel for
    for(unsigned int i = 0; i < A.size; i++){
        double sum = 0.0;

        for(unsigned int j = 0; j < A.nonnull_els_in_row; j++){
            struct ColVal *c = &(A.colvals[i * A.nonnull_els_in_row + j]);
            sum += c->val * x.vals[c->col];
        }
        res.vals[i] = sum;
    }
}

struct Vector omp_inv_diag_SpMV(const struct ELLMatrix A, const struct Vector x){
    assert(A.size == x.size);
    struct Vector ret = create_uninit_Vector(x.size);
    omp_inv_diag_SpMV_store(ret, A, x);
    return ret;
}

void omp_inv_diag_SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x){
    assert(res.size == A.size && A.size == x.size);

    #pragma omp parallel for
    for(unsigned int i = 0; i < A.size; i++){
        struct ColVal cv = A.colvals[A.diag_indices[i]];
        assert(cv.val);
        res.vals[i] = 1 / cv.val * x.vals[i];
    }
}
