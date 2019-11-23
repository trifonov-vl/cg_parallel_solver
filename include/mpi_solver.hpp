#include <ell_utils.hpp>
#include <mpi.h>
#include <iostream>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <algorithm>

#define crash(str, code, id) exit(Crash(str, code, id)) // via exit define so static analyzer knows its an exit point
static int Crash(const char *fmt, const int &code, const int &id){
    std::cout << id << " failed with msg: " << fmt << std::endl;
    MPI_Abort(MPI_COMM_WORLD, code);
    return code;
}

void solve_with_mpi(unsigned int nx, unsigned int ny, unsigned int nz, 
    unsigned int px, unsigned int py, unsigned int pz,
    double tol, unsigned int maxit    
);

