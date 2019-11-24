#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <sstream>  

struct ColVal{
    int col;
    double val;
    ColVal(const int &column = -1, const double &value = 0.0) : col(column), val(value) {}
};

class ELLMatrix{
    std::vector<ColVal> colvals;
    std::vector<unsigned int> diag_indices;

    public:
    
    unsigned int size;
    unsigned int nonnull_els_in_row;
    
    ELLMatrix(const unsigned int &size_, const unsigned int &nonnull_elems) : 
        colvals(size_ * nonnull_elems), 
        diag_indices(size_),
        size(size_), 
        nonnull_els_in_row(nonnull_elems){
            // if (nonnull_els_in_row > size)
            //     throw std::logic_error("Bad size arguments in ELL constructor");
        }
    
    ColVal get_elem_from_row(const unsigned int &row_idx, const unsigned int &elem_num) const {
        if (row_idx >= size || elem_num >= nonnull_els_in_row)
            throw std::logic_error("Bad input arguments in get_elem_from_row");
        
        return colvals[row_idx * nonnull_els_in_row + elem_num];
    }

    ColVal get_diagonal_element(const unsigned int &row_idx) const{
        if (row_idx >= size)
            throw std::logic_error("Bad input arguments in get_diagonal_element");
        
        return colvals[diag_indices[row_idx]];
    }

    void write_row(const unsigned int &row_idx, const std::vector<ColVal> &row) {
        if(row_idx >= size)
            throw std::logic_error("Bad input arguments in write_row");

        if(row.size() != nonnull_els_in_row)
            throw std::logic_error("Provided row do not align to this matrix");
        
        for(unsigned int i = 0; i < nonnull_els_in_row; i++) {
            const ColVal &cur = row[i];
            // check commented due to mpi realisation   
            // if(cur.col >= (int)size)
            //     throw std::logic_error("Provided row contains element with column more than size");
            colvals[row_idx * nonnull_els_in_row + i] = cur;

            if(cur.col >= 0 && (unsigned int)cur.col == row_idx){
                change_diag_indice(row_idx, row_idx * nonnull_els_in_row + i);
            }
        }
    }

    void change_diag_indice(const unsigned int &row_idx, const unsigned int &new_indice) {
        diag_indices[row_idx] = new_indice;
    }
    friend std::ostream & operator << (std::ostream &out, const ELLMatrix &m);

    std::string to_string() const;
};

std::vector<ColVal> generate_row_3D_DECART(
    const unsigned int &i, const unsigned int &j, const unsigned int &k, 
    const unsigned int &Nx, const unsigned int &Ny, const unsigned int &Nz);
