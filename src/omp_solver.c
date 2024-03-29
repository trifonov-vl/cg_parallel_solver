#include <omp_solver.h>

int omp_solve(const struct ELLMatrix A, const struct Vector b, const struct Vector x, double eps, int max_it){
    assert(A.size == b.size && b.size == x.size);
    assert(eps > 0 && max_it >= 0);
    
    double prev_ro, ro, alpha;
    struct Vector tmp = omp_SpMV(A, x);
    struct Vector r = omp_axpby(1.0, b, -1.0, tmp);
    delete_Vector(&tmp);
    
    struct Vector z = create_uninit_Vector(x.size);
    struct Vector q = create_uninit_Vector(x.size);
    struct Vector p = create_uninit_Vector(x.size);

    int it_num = 0;
    for(; it_num < max_it; it_num++){
        omp_inv_diag_SpMV_store(z, A, r);
        
        ro = omp_dot(r, z);
        if(it_num > 0){
            assert(prev_ro);
            omp_axpby_store(p, 1.0, z, ro / prev_ro, p);
        }
        else{
            copy_from_Vector_to_Vector(z, p);
        }
        prev_ro = ro;

        omp_SpMV_store(q, A, p);
        alpha = ro / omp_dot(p, q);
        omp_axpby_store(x, 1.0, x, alpha, p);
        omp_axpby_store(r, 1.0, r, -alpha, q);
        
        if(ro < eps){
            break;
        }
    }

    delete_Vector(&r);
    delete_Vector(&z);
    delete_Vector(&q);
    delete_Vector(&p);
    
    return it_num + 1;
}