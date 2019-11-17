#include <argp.h>
#include <stdarg.h>
#include <stdlib.h>
#include <ell_utils.h>
#include <vector_utils.h>
#include <mpi.h>

char doc[] = "CG parallel MPI solver";

static struct argp_option options[] = {
    { "nx", 'x', "uint", 0, "x matrix dimension size", 0 },
    { "ny", 'y', "uint", 0, "y matrix dimension size", 0 },
    { "nz", 'z', "uint", 0, "z matrix dimension size", 0 },
    { "px", 'a', "uint", 0, "px number of subregions", 0 },
    { "py", 'b', "uint", 0, "py number of subregions", 0 },
    { "pz", 'c', "uint", 0, "pz numeber of subregions", 0 },
    { "tol", 't', "double", 0, "residual", 0 },
    { "maxit", 'm', "uint", 0, "max iteration number", 0 },
    { 0 }
};

struct arguments {
    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
    unsigned int px;
    unsigned int py;
    unsigned int pz;
    double tol;
    double maxit;
};

static error_t parse_option(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;
    
    switch (key) {
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
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

struct argp argp = { options, parse_option, 0, doc, 0, 0, 0 };

struct arguments arguments;
int mpi_initialized = 0; // flag that MPI is initialized
int MyID = 0; // process ID
int NumProc = 1; // number of processes 
int MASTER_ID = 0; // master process ID
MPI_Comm MCW = MPI_COMM_WORLD; // default communicator

#define crash(...) exit(Crash( __VA_ARGS__)) // via exit define so static analyzer knows its an exit point
static int Crash(const char *fmt,...){ // termination of program due to error
    va_list ap;
    if(mpi_initialized) fprintf(stderr,"\nEpic fail: MyID = %d\n",MyID);
    else fprintf(stderr,"\nEpic fail: \n");

    va_start(ap,fmt);
    vfprintf(stderr,fmt,ap);
    va_end(ap);

    fprintf(stderr,"\n");
    fflush(stderr);
    
    if(mpi_initialized) MPI_Abort(MPI_COMM_WORLD,-1);
    return 0;
}

int printf0(const char *fmt,...){ // Write to stdout from Master process
    int r = 0;
    va_list ap;
    if(MyID==MASTER_ID){
        va_start(ap,fmt);  
        r=vfprintf(stdout,fmt,ap);  
        va_end(ap);
    }
    fflush(stdout);
    return(r);
}

void proc_id2decart_id(unsigned int *decart_x, unsigned int *decart_y, unsigned int *decart_z){
    *decart_x = MyID % arguments.px;
    *decart_y = MyID / arguments.px % arguments.py;
    *decart_z = MyID / (arguments.px * arguments.py) % arguments.pz;
}

int decart_id2proc_id(unsigned int decart_x, unsigned int decart_y, unsigned int decart_z){
    return decart_x + arguments.px * decart_y + arguments.px * arguments.py * decart_z;
}

int global2proc_id(unsigned int x, unsigned int y, unsigned int z){
    unsigned int decart_x = x / arguments.px;
    unsigned int decart_y = y / arguments.py;
    unsigned int decart_z = z / arguments.pz;
    return decart_id2proc_id(decart_x, decart_y, decart_z);
}

void proc_global_region_boundaries(unsigned int res[]){
    unsigned int x_id, y_id, z_id;
    proc_id2decart_id(&x_id, &y_id, &z_id);

    unsigned int N_per_x = arguments.nx / arguments.px;
    unsigned int N_per_y = arguments.ny / arguments.py;
    unsigned int N_per_z = arguments.nz / arguments.pz;

    res[0] = N_per_x * x_id; res[1] = N_per_x * (x_id + 1);
    res[2] = N_per_y * y_id; res[3] = N_per_y * (y_id + 1);
    res[4] = N_per_z * z_id; res[5] = N_per_z * (z_id + 1);
}



void create_part(){
    int proc_region_boundaries[6];
    proc_global_region_boundaries(proc_region_boundaries);
    unsigned int x_local_size = proc_region_boundaries[1] - proc_region_boundaries[0];
    unsigned int y_local_size = proc_region_boundaries[3] - proc_region_boundaries[2];
    unsigned int z_local_size = proc_region_boundaries[5] - proc_region_boundaries[4];
    unsigned int local_region_size = x_local_size * y_local_size * z_local_size;
    unsigned int extended_region_size = local_region_size;

    if (proc_region_boundaries[0] != 0){
        extended_region_size += y_local_size * z_local_size;
    }
    if (proc_region_boundaries[1] < arguments.nx){
        extended_region_size += y_local_size * z_local_size;
    }
    if (proc_region_boundaries[2] != 0){
        extended_region_size += x_local_size * z_local_size;
    }
    if (proc_region_boundaries[3] < arguments.ny){
        extended_region_size += x_local_size * z_local_size;
    }
    if (proc_region_boundaries[4] != 0){
        extended_region_size += y_local_size * x_local_size;
    }
    if (proc_region_boundaries[5] < arguments.nz){
        extended_region_size += y_local_size * x_local_size;
    }

    int *part = malloc(sizeof(int) * extended_region_size);
    unsigned int *L2G = malloc(sizeof(unsigned int) * extended_region_size);
    
    unsigned int idx = 0;
    for(unsigned int x = proc_region_boundaries[0]; x < proc_region_boundaries[1]; x++){
        for(unsigned int y = proc_region_boundaries[2]; y < proc_region_boundaries[3]; y++){
            for(unsigned int z = proc_region_boundaries[4]; z < proc_region_boundaries[5]; z++){
                L2G[idx] = x + y * arguments.nx + z * arguments.nx*arguments.ny;
                part[idx] = MyID;
                idx++;
            }
        }
    }

    if (proc_region_boundaries[0] != 0){
        for(unsigned int y = proc_region_boundaries[2]; y < proc_region_boundaries[3]; y++){
            for(unsigned int z = proc_region_boundaries[4]; z < proc_region_boundaries[5]; z++){
                L2G[idx] = proc_region_boundaries[0] - 1 + y * arguments.nx + z * arguments.nx*arguments.ny;
                part[idx] = global2proc_id(proc_region_boundaries[0] - 1, y, z);
                idx++;
            }
        }
    }
    if (proc_region_boundaries[1] < arguments.nx){
        for(unsigned int y = proc_region_boundaries[2]; y < proc_region_boundaries[3]; y++){
            for(unsigned int z = proc_region_boundaries[4]; z < proc_region_boundaries[5]; z++){
                L2G[idx] = proc_region_boundaries[1] + y * arguments.nx + z * arguments.nx*arguments.ny;
                part[idx] = global2proc_id(proc_region_boundaries[1], y, z);
                idx++;
            }
        }
    }
    if (proc_region_boundaries[2] != 0){
        for(unsigned int x = proc_region_boundaries[0]; x < proc_region_boundaries[1]; x++){
            for(unsigned int z = proc_region_boundaries[4]; z < proc_region_boundaries[5]; z++){
                L2G[idx] = x + (proc_region_boundaries[2] - 1) * arguments.nx + z * arguments.nx*arguments.ny;
                part[idx] = global2proc_id(x, proc_region_boundaries[2] - 1, z);
                idx++;
            }
        }
    }
    if (proc_region_boundaries[3] < arguments.ny){
        for(unsigned int x = proc_region_boundaries[0]; x < proc_region_boundaries[1]; x++){
            for(unsigned int z = proc_region_boundaries[4]; z < proc_region_boundaries[5]; z++){
                L2G[idx] = x + proc_region_boundaries[3] * arguments.nx + z * arguments.nx*arguments.ny;
                part[idx] = global2proc_id(x, proc_region_boundaries[3], z);
                idx++;
            }
        }
    }
    if (proc_region_boundaries[4] != 0){
        for(unsigned int x = proc_region_boundaries[0]; x < proc_region_boundaries[1]; x++){
            for(unsigned int y = proc_region_boundaries[2]; y < proc_region_boundaries[3]; y++){
                L2G[idx] = x + y * arguments.nx + (proc_region_boundaries[4] - 1) * arguments.nx*arguments.ny;
                part[idx] = global2proc_id(x, y, (proc_region_boundaries[4] - 1));
                idx++;
            }
        }
    }
    if (proc_region_boundaries[5] < arguments.nz){
        for(unsigned int x = proc_region_boundaries[0]; x < proc_region_boundaries[1]; x++){
            for(unsigned int y = proc_region_boundaries[2]; y < proc_region_boundaries[3]; y++){
                L2G[idx] = x + y * arguments.nx + proc_region_boundaries[5] * arguments.nx*arguments.ny;
                part[idx] = global2proc_id(x, y, proc_region_boundaries[5]);
                idx++;
            }
        }
    }



    



}

int main(int argc, char *argv[]) {
    int mpi_res; 

    mpi_res = MPI_Init(&argc, &argv);
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Init failed (code %d)\n", mpi_res);
    
    arguments.nx = 0;
    arguments.ny = 0;
    arguments.nz = 0;
    arguments.maxit = 0;
    arguments.px = 0;
    arguments.py = 0;
    arguments.pz = 0;

    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Cart_create failed (code %d)\n", mpi_res);

    mpi_res = MPI_Comm_rank(MCW,&MyID); 
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_rank failed (code %d)\n", mpi_res);

    mpi_res = MPI_Comm_size(MCW,&NumProc);
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_size failed (code %d)\n", mpi_res);
    


    
    
    run_solvers(arguments.nx, arguments.ny, arguments.nz, arguments.px, arguments.py, arguments.pz, arguments.nseeds, arguments.maxit, arguments.tol);
    return 0;
}