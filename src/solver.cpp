#include <solver.hpp>

int solve(
    const ELLMatrix &A, const std::vector<double> &b, 
    std::vector<double> &x, const double &eps, const int &max_it){

    if(A.size != b.size() || b.size() != x.size())
        throw std::logic_error("Vectors and matrices must have equal sizes");
    
    if(eps <= 0 || max_it < 0)
        throw std::logic_error("Solvers bad params");
    
    std::vector<double> tmp = SpMV(A, x);
    std::vector<double> r = axpby(1.0, b, -1.0, tmp);

    std::vector<double> z(A.size);
    std::vector<double> q(A.size);
    std::vector<double> p(A.size);

    int it_num = 0;
    double prev_ro;
    
    for(; it_num < max_it; it_num++){
        inv_diag_SpMV_store(z, A, r);
        
        double ro = dot(r, z);
        if(it_num > 0){
            axpby_store(p, 1.0, z, ro / prev_ro, p);
        }
        else{
            z = p;
        }
        prev_ro = ro;
        
        SpMV_store(q, A, p);
        double alpha = ro / dot(p, q);
        axpby_store(x, 1.0, x, alpha, p);
        axpby_store(r, 1.0, r, -alpha, q);
        
        if(ro < eps){
            break;
        }
    }
    return it_num + 1;
}