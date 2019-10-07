#include <argp.h>
#include <stdlib.h>
#include <ell_utils.h>
#include <vector_utils.h>
#include <solver.h>

char doc[] = "CG parallel solver";

static struct argp_option options[] = {
    { "qa", 'q', 0, 0, "Test base operations", 0 },
    { "nx", 'x', "uint", 0, "x matrix dimension size", 0 },
    { "ny", 'y', "uint", 0, "y matrix dimension size", 0 },
    { "nz", 'z', "uint", 0, "z matrix dimension size", 0 },
    { "tol", 't', "double", 0, "residual", 0 },
    { "maxit", 'm', "uint", 0, "max iteration number", 0 },
    { "nt", 'n', "uint", 0, "parallel thread number", 0 },  
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
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

struct argp argp = { options, parse_option, 0, doc, 0, 0, 0 };

int main(int argc, char *argv[]) {

    struct arguments arguments;
    arguments.qa = 0;
    arguments.nx = 0;
    arguments.ny = 0;
    arguments.nz = 0;
    arguments.maxit = 0;
    arguments.nt = 1;

    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    
    struct ELLMatrix m = generate_ELL_3D_DECART(arguments.nx, arguments.ny, arguments.nz);
    struct Vector b = create_cosine_Vector(m.size);
    struct Vector x = create_const_Vector(0, m.size);

    int it = solve(m, b, x, arguments.tol, arguments.maxit);
    printf("iters = %d, %f\n", it, compute_L2_norm(x));
    
    delete_ELL(&m);
    delete_Vector(&b);
    delete_Vector(&x);
    return 0;
}