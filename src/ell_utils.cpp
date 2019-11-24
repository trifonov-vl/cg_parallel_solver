#include <ell_utils.hpp>

std::string ELLMatrix::to_string() const{
    std::stringstream buffer;
    buffer << *this;
    return buffer.str();
}

std::ostream & operator << (std::ostream &out, const ELLMatrix &m){
    for(unsigned int i = 0; i < m.size; i++){
        int prev_col = 0;
        for(unsigned int j = 0; j < m.nonnull_els_in_row; j++){
            ColVal cur = m.get_elem_from_row(i, j);
            
            if(cur.col < 0){
                continue;
            }

            for(int k = prev_col; k < cur.col; k++)
                out << std::fixed << std::setprecision(2) << 0.0 << '\t';
            prev_col = cur.col + 1;
            out << std::fixed << std::setprecision(2) << cur.val << '\t';
        }
        for(unsigned int j = prev_col; j < m.size; j++)
            out << std::fixed << std::setprecision(2) << 0.0 << '\t';
        out << std::endl;
    }
    return out;
}


double generate_cosine(const int &i, const int &j){
    return cos(i * j + 3.14159265358979323846);
}


std::vector<ColVal> generate_row_3D_DECART(
    const unsigned int &i, const unsigned int &j, const unsigned int &k, 
    const unsigned int &Nx, const unsigned int &Ny, const unsigned int &Nz)
{
    
    std::vector<ColVal> ret(7);
    unsigned int matrix_row_idx = k * Nx * Ny + j * Nx + i;
    double ovrl_sum = 0.0;

    if(k > 0){
        int column = matrix_row_idx-Nx*Ny;
        ret[0].col = column;
        ret[0].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(ret[0].val);
    }
    if(j > 0){
        int column = matrix_row_idx-Nx;
        ret[1].col = column;
        ret[1].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(ret[1].val);
    }
    if(i > 0){
        int column = matrix_row_idx-1;
        ret[2].col = column;
        ret[2].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(ret[2].val);
    }
    if(i < Nx - 1){
        int column = matrix_row_idx + 1;
        ret[4].col = column;
        ret[4].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(ret[4].val);
    }
    if(j < Ny - 1){
        int column = matrix_row_idx + Nx;
        ret[5].col = column;
        ret[5].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(ret[5].val);
    }
    if(k < Nz - 1){
        int column = matrix_row_idx + Nx * Ny;
        ret[6].col = column;
        ret[6].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(ret[6].val);
    }

    // dominant diagonal element
    {
        int column = matrix_row_idx;
        ret[3].col = column;
        ret[3].val = 1.5 * ovrl_sum;
    }

    return ret;
}
