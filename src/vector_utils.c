#include <vector_utils.h>

struct Vector create_uninit_Vector(unsigned int size){
    assert(size);

    struct Vector ret;
    ret.size = size;
    
    ret.vals = malloc(sizeof(double) * size);
    if(!ret.vals){
        perror("Error in allocation");
    }

    return ret;
}


struct Vector create_const_Vector(double val, unsigned int size){
    struct Vector ret = create_uninit_Vector(size);
    
    for(unsigned int i = 0; i < size; i++){
        ret.vals[i] = val;
    }

    return ret;
}

struct Vector create_cosine_Vector(unsigned int size){
    struct Vector ret = create_uninit_Vector(size);
    
    for(unsigned int i = 0; i < size; i++){
        ret.vals[i] = cos(i * i);
    }

    return ret;
}

struct Vector create_sin_Vector(unsigned int size){
    struct Vector ret = create_uninit_Vector(size);
    
    for(unsigned int i = 0; i < size; i++){
        ret.vals[i] = sin(i * i);
    }

    return ret;
}

struct Vector copy_Vector(const struct Vector v){
    struct Vector ret = create_uninit_Vector(v.size);
    memcpy(ret.vals, v.vals, sizeof(double) * v.size);
    return ret;
}

void copy_from_Vector_to_Vector(const struct Vector from, const struct Vector to){
    assert(from.size == to.size);
    memcpy(to.vals, from.vals, sizeof(double) * from.size);
}

void delete_Vector(struct Vector *v){
    assert(v && v->vals);
    free(v->vals);
    v->vals = 0;
}

double compute_sum(const struct Vector v){
    double ret = 0.0;

    for(unsigned int i = 0; i < v.size; i++){
        ret += v.vals[i];
    }

    return ret;
}

double compute_L2_norm(const struct Vector v){
    double ret = 0.0;

    for(unsigned int i = 0; i < v.size; i++){
        ret += v.vals[i] * v.vals[i];
    }
    
    return sqrt(ret);
}

double compute_L1_norm(const struct Vector v){
    double ret = 0.0;

    for(unsigned int i = 0; i < v.size; i++){
        ret += fabs(v.vals[i]);
    }
    
    return ret;
}


double compute_L_inf_norm(const struct Vector v){
    double ret = 0.0;

    for(unsigned int i = 0; i < v.size; i++){
        ret = ret > fabs(v.vals[i]) ? ret : fabs(v.vals[i]);
    }
    
    return ret;
}
