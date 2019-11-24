#include <argp.h>
#include <cstdarg>
#include <cstdlib>
#include <mpi.h>
#include <iostream>
#include <mpi_solver.hpp>

char doc[] = "CG parallel MPI solver";

static struct argp_option options[] = {
    { "qa", 'q', 0, 0, "Test base operations", 0 },
    { "nx", 'x', "uint", 0, "x matrix dimension size", 0 },
    { "ny", 'y', "uint", 0, "y matrix dimension size", 0 },
    { "nz", 'z', "uint", 0, "z matrix dimension size", 0 },
    { "px", 'a', "uint", 0, "px number of subregions", 0 },
    { "py", 'b', "uint", 0, "py number of subregions", 0 },
    { "pz", 'c', "uint", 0, "pz numeber of subregions", 0 },
    { "tol", 't', "double", 0, "residual", 0 },
    { "maxit", 'm', "uint", 0, "max iteration number", 0 },
    { "nseeds", 's', "uint", 0, "number of seeds", 0},
    { 0 }
};

struct arguments {
    int qa;
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
    unsigned int px;
    unsigned int py;
    unsigned int pz;
    double tol;
    unsigned int maxit;
    unsigned int nseeds;
};

static error_t parse_option(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = (struct arguments*) state->input;
    
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
    case 'a':
        arguments->px = strtoul(arg, 0, 10);;
        break;
    case 'b':
        arguments->py = strtoul(arg, 0, 10);
        break;
    case 'c':
        arguments->pz = strtoul(arg, 0, 10);
        break;
    case 't':
        arguments->tol = strtod(arg, 0);
        break;
    case 'm':
        arguments->maxit = strtoul(arg, 0, 10);
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

int main(int argc, char *argv[]) {
    int mpi_res; 

    mpi_res = MPI_Init(&argc, &argv);
    if(mpi_res!= MPI_SUCCESS){
        std::cout << "MPI_Init failed" << std::endl;
        return mpi_res;
    }

    struct arguments arguments;
    arguments.qa = 0;
    arguments.nx = 0;
    arguments.ny = 0;
    arguments.nz = 0;
    arguments.maxit = 0;
    arguments.px = 0;
    arguments.py = 0;
    arguments.pz = 0;
    arguments.nseeds = 1;

    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    solve_with_mpi(arguments.nx, arguments.ny, arguments.nz,
        arguments.px, arguments.py, arguments.pz, arguments.tol, arguments.maxit, arguments.qa, arguments.nseeds
    );
    MPI_Finalize();
    
    return 0;
}