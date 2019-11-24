#include <mpi.h>
#include <iostream>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

#include <ell_utils.hpp>
#include <ops_utils.hpp>


void test_mpi_solver(unsigned int nx, unsigned int ny, unsigned int nz, 
    unsigned int px, unsigned int py, unsigned int pz,
    double tol, unsigned int maxit, bool qa, unsigned int nseeds
);

