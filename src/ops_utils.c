#include <ops_utils.h>

double dot(const struct Vector v1, const struct Vector v2){
    assert(v1.size == v2.size);
    double res = 0.0;

    for(unsigned int i = 0; i < v1.size; i++){
        res += v1.vals[i] * v2.vals[i];
    }

    return res;
}

void axpby_store(const struct Vector res, double a, const struct Vector x, double b, const struct Vector y){
    assert(x.size == y.size && y.size == res.size);
    
    for(unsigned int i = 0; i < x.size; i++){
        res.vals[i] = a * x.vals[i] + b * y.vals[i];
    }
}

struct Vector axpby(double a, const struct Vector x, double b, const struct Vector y){
    assert(x.size == y.size);
    struct Vector ret = create_uninit_Vector(x.size);
    axpby_store(ret, a, x, b, y);
    return ret;
}


void SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x){
    assert(res.vals != x.vals);
    assert(res.size == x.size && x.size == A.size);

    for(unsigned int i = 0; i < A.size; i++){
        double sum = 0.0;

        for(unsigned int j = 0; j < A.nonnull_els_in_row; j++){
            struct ColVal *c = &(A.colvals[i * A.nonnull_els_in_row + j]);
            sum += c->val * x.vals[c->col];
        }
        res.vals[i] = sum;
    }
}


struct Vector SpMV(const struct ELLMatrix A, const struct Vector x){
    assert(A.size == x.size);
    struct Vector ret = create_uninit_Vector(x.size);
    SpMV_store(ret, A, x);
    return ret;
}


void inv_diag_SpMV_store(const struct Vector res, const struct ELLMatrix A, const struct Vector x){
    assert(res.size == A.size && A.size == x.size);

    for(unsigned int i = 0; i < A.size; i++){
        assert(A.diag_indices[i]);
        res.vals[i] = 1 / A.diag_indices[i] * x.vals[i];
    }
}


struct Vector inv_diag_SpMV(const struct ELLMatrix A, const struct Vector x){
    assert(A.size == x.size);
    struct Vector ret = create_uninit_Vector(x.size);
    inv_diag_SpMV_store(ret, A, x);
    return ret;
}
