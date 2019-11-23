#include <mpi_solver.hpp>

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
    unsigned int decart_x = global.x / ceil(pars.Nx / pars.Px);
    unsigned int decart_y = global.y / ceil(pars.Ny / pars.Py);
    unsigned int decart_z = global.z / ceil(pars.Nz / pars.Pz);
    return (decart_x + pars.Px * decart_y + pars.Px * pars.Py * decart_z) % pars.ovrl_proc_num;
}

struct RegionBoundaries{
    unsigned int start_x, end_x, start_y, end_y, start_z, end_z;
    unsigned int x_size, y_size, z_size;

    RegionBoundaries(const Parameters &pars){
        unsigned int N_per_x = pars.Nx / pars.Px;
        unsigned int N_per_y = pars.Ny / pars.Py;
        unsigned int N_per_z = pars.Nz / pars.Pz;

        start_x = N_per_x * pars.decart_id.x; end_x = N_per_x * (pars.decart_id.x + 1);
        start_y = N_per_y * pars.decart_id.y; end_y = N_per_y * (pars.decart_id.y + 1);
        start_z = N_per_z * pars.decart_id.z; end_z = N_per_z * (pars.decart_id.z + 1);
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


void pprintf(const std::string &s, const int &proc_id, const int &proc_num){ 
    int r = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    for(int p=0; p<proc_num; p++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(proc_id != p) continue; 
        std::cout << "id: " << proc_id << " " << s << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
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

void solve_with_mpi(unsigned int nx, unsigned int ny, unsigned int nz, 
    unsigned int px, unsigned int py, unsigned int pz,
    double tol, unsigned int maxit){

    int mpi_res, current_proc_id, ovrl_proc_number;

    mpi_res = MPI_Comm_rank(MPI_COMM_WORLD, &current_proc_id); 
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_rank failed", mpi_res, current_proc_id);

    mpi_res = MPI_Comm_size(MPI_COMM_WORLD, &ovrl_proc_number);
    if(mpi_res!= MPI_SUCCESS) 
        crash("MPI_Comm_size failed", mpi_res, current_proc_id);

    Parameters pars(nx, ny, nz, px, py, pz, ovrl_proc_number);

    auto part_L2G = create_part_L2G(pars);
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

    ELLMatrix local_matrix = init_local_matrix(pars, part_L2G);
    pprintf("after local", pars.proc_id, pars.ovrl_proc_num);
    pprintf("\n" + matrix_columns_to_string(local_matrix, part_L2G), pars.proc_id, pars.ovrl_proc_num);


    auto inp_out = create_messaging_vectors(pars.proc_id, local_matrix, part_L2G);

    output_str = "Input nodes: \n";
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