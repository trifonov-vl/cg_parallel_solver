#include <argp.h>
#include <stdlib.h>
#include <ell_utils.h>
#include <vector_utils.h>
#include <solver.h>
#include <omp_solver.h>

char doc[] = "CG parallel solver";

static struct argp_option options[] = {
    { "qa", 'q', 0, 0, "Test base operations", 0 },
    { "nx", 'x', "uint", 0, "x matrix dimension size", 0 },
    { "ny", 'y', "uint", 0, "y matrix dimension size", 0 },
    { "nz", 'z', "uint", 0, "z matrix dimension size", 0 },
    { "tol", 't', "double", 0, "residual", 0 },
    { "maxit", 'm', "uint", 0, "max iteration number", 0 },
    { "nt", 'n', "uint", 0, "parallel thread number", 0 },
    { "nseeds", 's', "uint", 0, "number of seeds", 0},
    { 0 }
};

struct arguments {
    int qa;
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
    double tol;
    double maxit;
    unsigned int nt;
    unsigned int nseeds;
};

static error_t parse_option(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;
    
    switch (key) {
    case 'q':
        arguments->qa=1;
        break;
    case 'x':
        arguments->nx = strtoul(arg, 0, 10);;
        break;
    case 'y':
        arguments->ny = strtoul(arg, 0, 10);
        break;
    case 'z':
        arguments->nz = strtoul(arg, 0, 10);
        break;
    case 't':
        arguments->tol = strtod(arg, 0);
        break;
    case 'm':
        arguments->maxit = strtoul(arg, 0, 10);
        break;
    case 'n':
        arguments->nt = strtoul(arg, 0, 10);
        break;
    case 's':
        arguments->nseeds = strtoul(arg, 0, 10);
        break;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

struct argp argp = { options, parse_option, 0, doc, 0, 0, 0 };


void run_basic_operations_qa(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nseeds){
    unsigned int n = nx * ny * nz;
    struct Vector x = create_cosine_Vector(n);
    struct Vector y = create_sin_Vector(n);
    double t = 0.0;

    for(unsigned int i = 0; i < nseeds; i++){
        double start = omp_get_wtime();
        double res = dot(x, y);
        t += omp_get_wtime() - start;
        printf("DOT result = %15.12E\n", res);
    }   
    
    printf("DOT mean time = %5.3f s\n\n", t / nseeds);

    t = 0.0;
    for(unsigned int i = 0; i < nseeds; i++){
        double start = omp_get_wtime();
        double res = omp_dot(x, y);
        t += omp_get_wtime() - start;
        printf("OMP DOT result = %15.12E\n", res);
    }   
    
    printf("DOT OMP mean time = %5.3f s\n\n", t / nseeds);

    t = 0.0;
    for(unsigned int i = 0; i < nseeds; i++){
        double start = omp_get_wtime();
        struct Vector vec_res = axpby(2.0, x, 3.0, y);
        t += omp_get_wtime() - start;
        printf("AXPBY L2 result = %15.12E, L_inf result = %15.12E\n", compute_L2_norm(vec_res), compute_L_inf_norm(vec_res));
        delete_Vector(&vec_res);
    }   
    
    printf("AXPBY mean time = %5.3f s\n\n", t / nseeds);

    t = 0.0;
    for(unsigned int i = 0; i < nseeds; i++){
        double start = omp_get_wtime();
        struct Vector vec_res = omp_axpby(2.0, x, 3.0, y);
        t += omp_get_wtime() - start;
        printf("OMP AXPBY L2 result = %15.12E, L_inf result = %15.12E\n", compute_L2_norm(vec_res), compute_L_inf_norm(vec_res));
        delete_Vector(&vec_res);
    }   
    
    printf("OMP AXPBY mean time = %5.3f s\n\n", t / nseeds);

    struct ELLMatrix m = generate_ELL_3D_DECART(nx, ny, nz);
    
    t = 0.0;
    for(unsigned int i = 0; i < nseeds; i++){
        double start = omp_get_wtime();
        struct Vector vec_res = SpMV(m, x);
        t += omp_get_wtime() - start;
        printf("SPMV L2 result = %15.12E, L_inf result = %15.12E\n", compute_L2_norm(vec_res), compute_L_inf_norm(vec_res));
        delete_Vector(&vec_res);
    }   
    
    printf("SpMV mean time = %5.3f s\n\n", t / nseeds);

    t = 0.0;
    for(unsigned int i = 0; i < nseeds; i++){
        double start = omp_get_wtime();
        struct Vector vec_res = omp_SpMV(m, x);
        t += omp_get_wtime() - start;
        printf("OMP SPMV L2 result = %15.12E, L_inf result = %15.12E\n", compute_L2_norm(vec_res), compute_L_inf_norm(vec_res));
        delete_Vector(&vec_res);
    }   
    
    printf("OMP SpMV mean time = %5.3f s\n\n", t / nseeds);

    delete_ELL(&m);
    delete_Vector(&x);
    delete_Vector(&y);
}


void run_solvers(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int nseeds, unsigned int maxit, double tol){
    struct ELLMatrix m = generate_ELL_3D_DECART(nx, ny, nz);
    struct Vector b = create_cosine_Vector(m.size);
    double t = 0.0;

    for(unsigned int i = 0; i < nseeds; i++){
        struct Vector x = create_const_Vector(0, m.size);
        double start = omp_get_wtime();
        int it = solve(m, b, x, tol, maxit);
        t += omp_get_wtime() - start;
        struct Vector r = SpMV(m, x);
        axpby_store(r, 1.0, r, -1.0, b);
        printf("CG Solver iters = %d, residual norm / b norm = %15.12E\n", it, compute_L2_norm(r) / compute_L2_norm(b));
        delete_Vector(&r);
        delete_Vector(&x);
    }

    printf("CG Solver mean time = %5.3f\n\n", t / nseeds);

    t = 0.0;
    for(unsigned int i = 0; i < nseeds; i++){
        struct Vector x = create_const_Vector(0, m.size);
        double start = omp_get_wtime();
        int it = omp_solve(m, b, x, tol, maxit);
        t += omp_get_wtime() - start;
        struct Vector r = SpMV(m, x);
        axpby_store(r, 1.0, r, -1.0, b);
        printf("CG OMP Solver iters = %d, residual norm / b norm = %15.12E\n", it, compute_L2_norm(r) / compute_L2_norm(b));
        delete_Vector(&r);
        delete_Vector(&x);
    }

    printf("CG OMP Solver mean time = %5.3f\n\n", t / nseeds);
    
    delete_ELL(&m);
    delete_Vector(&b);
}

int main(int argc, char *argv[]) {

    struct arguments arguments;
    arguments.qa = 0;
    arguments.nx = 0;
    arguments.ny = 0;
    arguments.nz = 0;
    arguments.maxit = 0;
    arguments.nt = 1;
    arguments.nseeds = 1;

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    omp_set_num_threads(arguments.nt);
    if(arguments.qa){
        run_basic_operations_qa(arguments.nx, arguments.ny, arguments.nz, arguments.nseeds);
    }
    
    run_solvers(arguments.nx, arguments.ny, arguments.nz, arguments.nseeds, arguments.maxit, arguments.tol);
    return 0;
}