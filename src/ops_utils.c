#include <ops_utils.h>

double dot(const struct Vector v1, const struct Vector v2){
    assert(v1.size == v2.size);
    double res = 0.0;

    for(unsigned int i = 0; i < v1.size; i++){
        res += v1.vals[i] * v2.vals[i];
    }

    return res;
}

struct Vector axpby(double a, const struct Vector x, double b, const struct Vector y){
    assert(x.size == y.size);    
    struct Vector ret = create_uninit_Vector(x.size);

    for(unsigned int i = 0; i < x.size; i++){
        ret.vals[i] = a * x.vals[i] + b * y.vals[i];
    }

    return ret;
}

struct Vector SpMV(const struct ELLMatrix A, const struct Vector x){
    assert(A.size == x.size);
    struct Vector ret = create_uninit_Vector(x.size);

    for(unsigned int i = 0; i < A.size; i++){
        double sum = 0.0;

        for(unsigned int j = 0; j < A.nonnull_els_in_row; j++){
            struct ColVal *c = &(A.colvals[i * A.nonnull_els_in_row + j]);
            sum += c->val * x.vals[c->col];
        }
        ret.vals[i] = sum;
    }

    return ret;
}
