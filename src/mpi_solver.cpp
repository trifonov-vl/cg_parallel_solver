#include <mpi_solver.hpp>

#define DEBUG 1

#define crash(str, code, id) exit(Crash(str, code, id)) // via exit define so static analyzer knows its an exit point
static int Crash(const char *fmt, const int &code, const int &id){
    std::cout << id << " failed with msg: " << fmt << std::endl << std::flush;
    MPI_Abort(MPI_COMM_WORLD, code);
    return code;
}


void pprintf(const std::string &s, const int &proc_id, const int &proc_num){ 
    MPI_Barrier(MPI_COMM_WORLD);
    for(int p=0; p<proc_num; p++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(proc_id != p) continue; 
        std::cout << "id: " << proc_id << " " << s << std::endl << std::flush;
    }
    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
}

struct Coordinates{
    unsigned int x, y, z;

    Coordinates(unsigned int x_ = 0, unsigned int y_ = 0, unsigned int z_ = 0) : x(x_), y(y_), z(z_) {} 
};

struct Parameters{
    unsigned int Nx, Ny, Nz, Px, Py, Pz;
    int proc_id;
    int ovrl_proc_num;
    Coordinates decart_id;

    Parameters(unsigned int nx, unsigned int ny, unsigned int nz, 
        unsigned int px, unsigned int py, unsigned int pz, unsigned int ovrl_proc_num_) : 
        Nx(nx), Ny(ny), Nz(nz), 
        Px(px), Py(py), Pz(pz), ovrl_proc_num(ovrl_proc_num_){
        
        int mpi_res = MPI_Comm_rank(MPI_COMM_WORLD, &proc_id); 
        if(mpi_res!= MPI_SUCCESS) 
            crash("MPI_Comm_rank failed", mpi_res, proc_id);
        
        decart_id = Coordinates(
            proc_id % Px, 
            proc_id / Px % Py, 
            proc_id / (Px * Py) % Pz);
    }
};

int global2proc_id(const Coordinates &global, const Parameters &pars){
    unsigned int decart_x = global.x / (pars.Nx / pars.Px + (pars.Nx / pars.Px != 0));
    unsigned int decart_y = global.y / (pars.Ny / pars.Py + (pars.Ny / pars.Py != 0));
    unsigned int decart_z = global.z / (pars.Nz / pars.Pz + (pars.Nz / pars.Pz != 0));
    return (decart_x + pars.Px * decart_y + pars.Px * pars.Py * decart_z) % pars.ovrl_proc_num;
}

struct RegionBoundaries{
    unsigned int start_x, end_x, start_y, end_y, start_z, end_z;
    unsigned int x_size, y_size, z_size;

    RegionBoundaries(const Parameters &pars){
        unsigned int N_per_x = pars.Nx / pars.Px + (pars.Nx / pars.Px != 0);
        unsigned int N_per_y = pars.Ny / pars.Py + (pars.Ny / pars.Py != 0);
        unsigned int N_per_z = pars.Nz / pars.Pz + (pars.Nz / pars.Pz != 0);

        start_x = N_per_x * pars.decart_id.x; end_x = N_per_x * (pars.decart_id.x + 1) > pars.Nx ? pars.Nx : N_per_x * (pars.decart_id.x + 1);
        start_y = N_per_y * pars.decart_id.y; end_y = N_per_y * (pars.decart_id.y + 1) > pars.Ny ? pars.Ny : N_per_y * (pars.decart_id.y + 1);
        start_z = N_per_z * pars.decart_id.z; end_z = N_per_z * (pars.decart_id.z + 1) > pars.Nz ? pars.Nz : N_per_z * (pars.decart_id.z + 1);
        x_size = end_x - start_x;
        y_size = end_y - start_y;
        z_size = end_z - start_z;
    }
};


std::vector<std::pair<int, unsigned int>> create_part_L2G(const Parameters &pars){
    RegionBoundaries local_boundaries(pars);
    std::vector<std::pair<int, unsigned int>> part_L2G;

    for(int x = local_boundaries.start_x - 1; x <= (int)local_boundaries.end_x; x++){
        if(x < 0 || x >= (int)pars.Nx)
            continue;

        for(int y = local_boundaries.start_y - 1; y <= (int)local_boundaries.end_y; y++){
            if(y < 0 || y >= (int)pars.Ny)
                continue;

            for(int z = local_boundaries.start_z - 1; z <= (int)local_boundaries.end_z; z++){
                if(z < 0 || z >= (int)pars.Nz)
                    continue;

                part_L2G.push_back(std::make_pair(
                    global2proc_id(Coordinates(x, y, z), pars), 
                    x + y * pars.Nx + z * pars.Nx * pars.Ny));
            }
        }
    }

    std::sort(part_L2G.begin(), part_L2G.end(),  [pars] (const auto& lhs, const auto& rhs) {
        if(lhs.first == pars.proc_id){
            if(rhs.first == pars.proc_id)
                return lhs.second < rhs.second;
            return true;
        }
        if(rhs.first == pars.proc_id){
            if(lhs.first == pars.proc_id)
                return lhs.second < rhs.second;
            return false;
        }
        return lhs < rhs;
    });
    return part_L2G;
}

ELLMatrix init_local_matrix(
    const Parameters &pars,
    const std::vector<std::pair<int, unsigned int>> &part_L2G){
    
    std::unordered_map<unsigned int, unsigned int> G2L;
    unsigned int idx = 0;
    for (const auto &i : part_L2G){
        G2L[i.second] = idx;
        idx++;
    }

    RegionBoundaries local_boundaries(pars);
    ELLMatrix ret(local_boundaries.x_size * local_boundaries.y_size * local_boundaries.z_size, 7);
    
    for(unsigned int i = 0; i < ret.size; i++){
        const unsigned int &global_idx = part_L2G[i].second;

        unsigned int global_x = global_idx % pars.Nx;
        unsigned int global_y = global_idx / pars.Nx % pars.Ny;
        unsigned int global_z = global_idx / pars.Nx / pars.Ny % pars.Nz;

        // get row with generated global values for local element
        std::vector<ColVal> row = generate_row_3D_DECART(
            global_x, global_y, global_z, pars.Nx, pars.Ny, pars.Nz);
              
        // need to change global columns to local ones
        for(auto &cv: row){
            if(cv.col >= 0)
                cv.col = G2L[cv.col];
        }
                
        ret.write_row(i, row);
    }
    return ret;
}

std::vector<unsigned int> sort_with_respect(const std::unordered_set<unsigned int> &to_sort, 
    const std::vector<std::pair<int, unsigned int>> &part_L2G){
        
    std::vector<std::pair<unsigned int, unsigned int>> sorted_pairs(to_sort.size());
    unsigned int idx = 0;
    for(const auto &i: to_sort){
        sorted_pairs[idx] = std::make_pair(part_L2G[i].second, i);
        idx++;
    }
    std::sort(sorted_pairs.begin(), sorted_pairs.end());
    
    std::vector<unsigned int> ret(to_sort.size());
    idx = 0;
    for(const auto &p: sorted_pairs){
        ret[idx] = p.second;
        idx++;
    }
    
    return ret;
}


std::pair<
    std::unordered_map<int, std::vector<unsigned int>>, 
    std::unordered_map<int, std::vector<unsigned int>>
    > create_messaging_vectors(
        int proc_id,
        const ELLMatrix &m, const std::vector<std::pair<int, unsigned int>> &part_L2G
    ){

    std::unordered_map<int, std::unordered_set<unsigned int>> input, output;
        
    for(unsigned int row_idx = 0; row_idx < m.size; row_idx++){
        std::unordered_set<int> row_procs;
        
        for(unsigned int row_elem = 0; row_elem < m.nonnull_els_in_row; row_elem++){
            const auto &cv = m.get_elem_from_row(row_idx, row_elem);
            if(cv.col < 0) continue;

            const int &owner = part_L2G[cv.col].first;
            if(owner != proc_id){
                input[owner].insert(cv.col);
                row_procs.insert(owner);
            }
        }

        for(const auto& proc : row_procs)
            output[proc].insert(row_idx);
    }

    std::unordered_map<int, std::vector<unsigned int>> input_ret, output_ret;
    for(const auto &p: input){
        input_ret[p.first] = sort_with_respect(p.second, part_L2G);
    }
    for(const auto &p: output){
        output_ret[p.first] = sort_with_respect(p.second, part_L2G);
    }

    return std::make_pair(input_ret, output_ret);
}

std::string matrix_columns_to_string(const ELLMatrix &m, 
    const std::vector<std::pair<int, unsigned int>> &part_L2G){
    
    std::string ret;

    for(unsigned int i = 0; i < m.size; i++){
        ret += std::to_string(part_L2G[i].second) + ": ";
        for(unsigned int j = 0; j < m.nonnull_els_in_row; j++){
            const auto &cv = m.get_elem_from_row(i,j);
            if(cv.col < 0)
                continue;
            
            ret += std::to_string(part_L2G[cv.col].second) + "\t";
        }
        ret += "\n";
    }
    return ret;
}

double mpi_dot(const std::vector<double> &x, const std::vector<double> &y, const unsigned int &halo_offset, const int &proc_id){
    double result = std::inner_product(x.begin(), x.begin() + halo_offset, y.begin(), 0.0);
    
    int mpi_res = MPI_Allreduce(&result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(mpi_res != MPI_SUCCESS)
        crash("MPI_Allreduce in dot failed", mpi_res, proc_id);
    
    return result;
}

double mpi_compute_L2_norm(const std::vector<double> &x, const unsigned int &halo_offset, const int &proc_id){
    double result = mpi_dot(x, x, halo_offset, proc_id);
    return sqrt(result);
}


double mpi_compute_L_inf_norm(const std::vector<double> &x, const int &proc_id){
    double result = 0.0;
    for(const auto &el : x){
        result = result > fabs(el) ? result : fabs(el);
    }
    int mpi_res = MPI_Allreduce(&result, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(mpi_res != MPI_SUCCESS)
        crash("MPI_Allreduce in mpi_compute_L_inf failed", mpi_res, proc_id);
    return result;
}

std::vector<double> create_cosine_local(const std::vector<std::pair<int, unsigned int>> &part_L2G, const int &proc_id){
    std::vector<double> ret;

    for(const auto &i: part_L2G){
        if(i.first != proc_id)
            break;

        ret.push_back(cos(i.second * i.second));
    }
    return ret;
}

std::vector<double> create_sin_local(const std::vector<std::pair<int, unsigned int>> &part_L2G, const int &proc_id){
    std::vector<double> ret;

    for(const auto &i: part_L2G){
        if(i.first != proc_id)
            break;

        ret.push_back(sin(i.second * i.second));
    }
    return ret;
}

std::vector<double> create_cosine_extended(const std::vector<std::pair<int, unsigned int>> &part_L2G){
    std::vector<double> ret;

    for(const auto &i: part_L2G){
        ret.push_back(cos(i.second * i.second));
    }
    return ret;
}


std::string double_to_string(const double &v, const unsigned int &w, const unsigned int &pres, const bool &scientific = false){
    std::stringstream buffer;
    if(scientific)
        buffer << std::scientific;
    else
        buffer << std::fixed;
    buffer << std::setw(w) << std::setprecision(pres) << v;
    return buffer.str();
}


void update_halo(
    const int &proc_id,
    std::vector<double> &x,
    const std::unordered_map<int, std::vector<unsigned int>> &input, 
    const std::unordered_map<int, std::vector<unsigned int>> &output,
    std::unordered_map<int, std::vector<double>> &input_buffers,
    std::unordered_map<int, std::vector<double>> &output_buffers
    ){
    
    std::vector<MPI_Request> requests;
    MPI_Request req;
    int mpi_res;
    for(const auto &p : input){
        mpi_res = MPI_Irecv(input_buffers[p.first].data(), p.second.size(), MPI_DOUBLE, p.first, 0, MPI_COMM_WORLD, &req);
        if(mpi_res != MPI_SUCCESS)
            crash("MPI_Irecv in halo_update failed", mpi_res, proc_id);
        requests.push_back(req);
    }
    for(const auto &p : output){
        unsigned int idx = 0;
        for(const auto &idx_for_send : p.second)
            output_buffers[p.first][idx++] = x[idx_for_send];
        
        mpi_res = MPI_Isend(output_buffers[p.first].data(), p.second.size(), MPI_DOUBLE, p.first, 0, MPI_COMM_WORLD, &req);
        if(mpi_res != MPI_SUCCESS)
            crash("MPI_Isend in halo_update failed", mpi_res, proc_id);
        requests.push_back(req);
    }

    mpi_res = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    if(mpi_res != MPI_SUCCESS)
        crash("MPI_Waitall in halo_update failed", mpi_res, proc_id);
    
    for(const auto &p : input){
        unsigned int idx = 0;
        for(const auto &idx_for_recv : p.second)
            x[idx_for_recv] = input_buffers[p.first][idx++];
    }
}



void run_qa(
    const Parameters &pars, unsigned int nseeds, const std::vector<std::pair<int, unsigned int>> &part_L2G, const ELLMatrix &local_matrix,
    const std::unordered_map<int, std::vector<unsigned int>> &input, const std::unordered_map<int, std::vector<unsigned int>> &output,
    std::unordered_map<int, std::vector<double>> &input_buffers, std::unordered_map<int, std::vector<double>> &output_buffers  
    ){
    
    auto x = create_cosine_local(part_L2G, pars.proc_id);
    auto y = create_sin_local(part_L2G, pars.proc_id);
    double t = 0.0;

    for(unsigned int i = 0; i < nseeds; i++){
        double start = MPI_Wtime();
        double res = mpi_dot(x, y, x.size(), pars.proc_id);
        t += MPI_Wtime() - start;
        pprintf("DOT result = " + double_to_string(res, 15, 12, true), pars.proc_id, pars.ovrl_proc_num);
    }

    if(pars.proc_id == 0){
        std::cout << "DOT mean time = " << double_to_string(t / nseeds, 5, 3, false) << std::endl << std::flush;
    }

    t = 0.0;
    for(unsigned int i = 0; i < nseeds; i++){
        double start = MPI_Wtime();
        auto vec_res = axpby(2.0, x, 3.0, y);
        double L2 = mpi_compute_L2_norm(vec_res, x.size(), pars.proc_id);
        double Linf = mpi_compute_L_inf_norm(vec_res, pars.proc_id);
        t += MPI_Wtime() - start;
        pprintf("AXPBY L2 result = " + double_to_string(L2, 15, 12, true) + 
            ", L_inf result = " + double_to_string(Linf, 15, 12, true), 
            pars.proc_id, pars.ovrl_proc_num);
    }

    if(pars.proc_id == 0){
        std::cout << "AXPBY mean time = " << double_to_string(t / nseeds, 5, 3, false) << std::endl << std::flush;
    }

    t = 0.0;
    x = create_cosine_extended(part_L2G);
    for(unsigned int i = 0; i < nseeds; i++){
        double start = MPI_Wtime();
        update_halo(pars.proc_id, x, input, output, input_buffers, output_buffers);
        auto vec_res = SpMV(local_matrix, x);
        t += MPI_Wtime() - start;
        pprintf("SPMV L2 result = " + double_to_string(mpi_compute_L2_norm(vec_res, local_matrix.size, pars.proc_id), 15, 12, true) + 
            ", L_inf result = " + double_to_string(mpi_compute_L_inf_norm(vec_res, pars.proc_id), 15, 12, true), 
            pars.proc_id, pars.ovrl_proc_num);
    }
    if(pars.proc_id == 0){
        std::cout << "SpMV mean time = " << double_to_string(t / nseeds, 5, 3, false) << std::endl << std::flush;
    }
}


int solve(
    const int &proc_id,
    const ELLMatrix &local_matrix, 
    const std::vector<double> &b, 
    std::vector<double> &x, 
    const std::unordered_map<int, std::vector<unsigned int>> &input, const std::unordered_map<int, std::vector<unsigned int>> &output,
    std::unordered_map<int, std::vector<double>> &input_buffers, std::unordered_map<int, std::vector<double>> &output_buffers,
    const double &eps, const int &max_it){
    
    std::vector<double> r = axpby(1.0, b, -1.0, SpMV(local_matrix, x));
    std::vector<double> z(x.size(), 0);
    std::vector<double> q(x.size(), 0);
    std::vector<double> p(x.size(), 0);

    int it_num = 0;
    double prev_ro;
    
    for(; it_num < max_it; it_num++){
        inv_diag_SpMV_store(z, local_matrix, r);
        
        double ro = mpi_dot(r, z, local_matrix.size, proc_id);
        if(it_num > 0){
            axpby_store(p, 1.0, z, ro / prev_ro, p);
        }
        else{
            p = z;
        }
        prev_ro = ro;
        
        update_halo(proc_id, p, input, output, input_buffers, output_buffers);
        SpMV_store(q, local_matrix, p);
        double alpha = ro / mpi_dot(p, q, local_matrix.size, proc_id);
        axpby_store(x, 1.0, x, alpha, p);
        axpby_store(r, 1.0, r, -alpha, q);
        
        if(ro < eps){
            break;
        }
    }
    return it_num + 1;
}

void run_solver(
    const Parameters &pars, unsigned int nseeds, const std::vector<std::pair<int, unsigned int>> &part_L2G, const ELLMatrix &local_matrix,
    const std::unordered_map<int, std::vector<unsigned int>> &input, const std::unordered_map<int, std::vector<unsigned int>> &output,
    std::unordered_map<int, std::vector<double>> &input_buffers, std::unordered_map<int, std::vector<double>> &output_buffers,
    const double &eps, const unsigned int max_it
    ){
    
    auto b = create_cosine_extended(part_L2G);
    double t = 0.0;

    for(unsigned int i = 0; i < nseeds; i++){
        std::vector<double> x(b.size(), 0);
        double start = MPI_Wtime();
        
        int it = solve(pars.proc_id, local_matrix, b, x, 
            input, output, input_buffers, output_buffers, eps, max_it);
        t += MPI_Wtime() - start;
        auto res = SpMV(local_matrix, x);
        axpby_store(res, 1.0, res, -1.0, b);
        pprintf("CG Solver iters = " + std::to_string(it) + 
            ", residual norm / b norm = " + 
            double_to_string(mpi_compute_L2_norm(res, local_matrix.size, pars.proc_id) / mpi_compute_L2_norm(b, local_matrix.size, pars.proc_id), 15, 12, true), 
            pars.proc_id, pars.ovrl_proc_num);
    }

    if(pars.proc_id == 0){
        std::cout << "CG solver mean time = " << double_to_string(t / nseeds, 5, 3, false) << std::endl << std::flush;
    }
}

void solve_with_mpi(unsigned int nx, unsigned int ny, unsigned int nz, 
    unsigned int px, unsigned int py, unsigned int pz,
    double tol, unsigned int maxit, bool qa, unsigned int nseeds){

    int mpi_res, current_proc_id, ovrl_proc_number;

    mpi_res = MPI_Comm_rank(MPI_COMM_WORLD, &current_proc_id); 
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_rank failed", mpi_res, current_proc_id);

    mpi_res = MPI_Comm_size(MPI_COMM_WORLD, &ovrl_proc_number);
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_size failed", mpi_res, current_proc_id);

    if(px*py*pz != (unsigned int)ovrl_proc_number)
        crash("px * py * pz != mpi proc number", 1, current_proc_id);

    Parameters pars(nx, ny, nz, px, py, pz, ovrl_proc_number);

    auto part_L2G = create_part_L2G(pars);

    if(DEBUG){
        pprintf("after part L2G", pars.proc_id, pars.ovrl_proc_num);

        std::string output_str = "Part nodes: \n";
        unsigned int idx = 0;
        for(const auto &pair : part_L2G){
            output_str += std::to_string(idx);
            output_str += ": ";
            output_str += std::to_string(pair.first);
            output_str += "\n";
            idx++;
        }
        pprintf(output_str, pars.proc_id, pars.ovrl_proc_num);

        output_str = "L2G nodes: \n";
        idx = 0;
        for(const auto &pair : part_L2G){
            output_str += std::to_string(idx);
            output_str += ": ";
            output_str += std::to_string(pair.second);
            output_str += "\n";
            idx++;
        }
        pprintf(output_str, pars.proc_id, pars.ovrl_proc_num);
    }

    ELLMatrix local_matrix = init_local_matrix(pars, part_L2G);
    
    if(DEBUG){
        pprintf("after local", pars.proc_id, pars.ovrl_proc_num);
        pprintf("\n" + matrix_columns_to_string(local_matrix, part_L2G), pars.proc_id, pars.ovrl_proc_num);
    }

    auto inp_out = create_messaging_vectors(pars.proc_id, local_matrix, part_L2G);

    if(DEBUG){
        std::string output_str = "Input nodes: \n";
        for(const auto &pair : inp_out.first){
            output_str += std::to_string(pair.first);
            output_str += ": ";
            for(const auto &n : pair.second){
                output_str += std::to_string(part_L2G[n].second);
                output_str += " ";
            }
            output_str += "\n";
        }
        pprintf(output_str, pars.proc_id, pars.ovrl_proc_num);
    
        output_str = "Output nodes: \n";
        for(const auto &pair : inp_out.second){
            output_str += std::to_string(pair.first);
            output_str += ": ";
            for(const auto &n : pair.second){
                output_str += std::to_string(part_L2G[n].second);
                output_str += " ";
            }
            output_str += "\n";
        }

        pprintf(output_str, pars.proc_id, pars.ovrl_proc_num);
    }

    std::unordered_map<int, std::vector<double>> input_buffers, output_buffers;

    for(const auto &p: inp_out.first){
        input_buffers[p.first] = std::vector<double>(p.second.size());
    }
    for(const auto &p: inp_out.second){
        output_buffers[p.first] = std::vector<double>(p.second.size());
    }

    if(qa){
        run_qa(pars, nseeds, part_L2G, local_matrix, inp_out.first, inp_out.second, input_buffers, output_buffers);
    }
    run_solver(pars, nseeds, part_L2G, local_matrix, inp_out.first, inp_out.second, input_buffers, output_buffers, tol, maxit);
}