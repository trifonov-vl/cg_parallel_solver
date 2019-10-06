#include <ell_utils.h>

void init_ELL(struct ELLMatrix *m, unsigned int size, unsigned int nonnull_els_in_row){
    assert(m);
    assert(size > 0 && nonnull_els_in_row > 0 && nonnull_els_in_row <= size);
    m->size = size;
    m->nonnull_els_in_row = nonnull_els_in_row;
    m->colvals = malloc(size * nonnull_els_in_row * sizeof(struct ColVal));
    if(!m->colvals){
        perror("Error in allocation");
    }
    m->diag_indices = malloc(size * sizeof(unsigned int));
    if(!m->diag_indices){
        perror("Error in allocation");
    }
}

void delete_ELL(struct ELLMatrix *m){
    assert(m && m->colvals);
    free(m->colvals);
    m->colvals = 0;
    free(m->diag_indices);
    m->diag_indices = 0;
}

void print_ELL(const struct ELLMatrix m){
    for(unsigned int i = 0; i < m.size; i++){
        int prev_col = 0;
        for(unsigned int j = 0; j < m.nonnull_els_in_row; j++){
            struct ColVal cur = m.colvals[i * m.nonnull_els_in_row + j];
            
            if(cur.col < 0){
                continue;
            }

            for(int k = prev_col; k < cur.col; k++)
                printf("%.3f\t", 0.0f);
            prev_col = cur.col + 1;
            printf("%.3f\t", cur.val);
        }
        for(unsigned int j = prev_col; j < m.size; j++)
            printf("%.3f\t", 0.0f);
        printf("\n");
    }
}


void writeRow(const struct ELLMatrix m, unsigned int row_idx, struct ColVal *colvals){
    assert(m.colvals && colvals);
    assert(row_idx < m.size);

    for(unsigned int i = 0; i < m.nonnull_els_in_row; i++){
        assert(colvals[i].col < 0 || (unsigned int)colvals[i].col < m.size);
        m.colvals[row_idx * m.nonnull_els_in_row + i] = colvals[i];

        if(colvals[i].col >= 0 && (unsigned int)colvals[i].col == row_idx){
            m.diag_indices[i] = row_idx * m.nonnull_els_in_row + i;
        }
    }
}

double generate_cosine(int i, int j){
    return cos(i * j + 3.14159265358979323846);
}

void generate_row_3D_DECART(struct ColVal *p, unsigned int i, unsigned int j, unsigned int k, unsigned int Nx, unsigned int Ny, unsigned int Nz){
    assert(p);
    unsigned int matrix_row_idx = k * Nx * Ny + j * Nx + i;
    double ovrl_sum = 0.0;

    for(int tmp = 0; tmp < 7; tmp++){
        p[tmp].col = -1;
        p[tmp].val = 0.0;
    }

    if(k > 0){
        int column = matrix_row_idx-Nx*Ny;
        p[0].col = column;
        p[0].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(p[0].val);
    }
    if(j > 0){
        int column = matrix_row_idx-Nx;
        p[1].col = column;
        p[1].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(p[1].val);
    }
    if(i > 0){
        int column = matrix_row_idx-1;
        p[2].col = column;
        p[2].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(p[2].val);
    }
    if(i < Nx - 1){
        int column = matrix_row_idx + 1;
        p[4].col = column;
        p[4].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(p[4].val);
    }
    if(j < Ny - 1){
        int column = matrix_row_idx + Nx;
        p[5].col = column;
        p[5].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(p[5].val);
    }
    if(k < Nz - 1){
        int column = matrix_row_idx + Nx * Ny;
        p[6].col = column;
        p[6].val = generate_cosine(matrix_row_idx, column);
        ovrl_sum += fabs(p[6].val);
    }

    // dominant diagonal element
    {
        int column = matrix_row_idx;
        p[3].col = column;
        p[3].val = 1.5 * ovrl_sum;
    }
}


struct ELLMatrix generate_ELL_3D_DECART(unsigned int Nx, unsigned int Ny, unsigned int Nz){
    struct ELLMatrix ret;
    struct ColVal tmp[7];
    init_ELL(&ret, Nx * Ny * Nz, 7);
    
    for(unsigned int i = 0; i < Nx; i++){
        for(unsigned int j = 0; j < Ny; j++){
            for(unsigned int k = 0; k < Nz; k++){
                generate_row_3D_DECART(tmp, i, j, k, Nx, Ny, Nz);
                writeRow(ret, k * Nx * Ny + j * Nx + i, tmp);
            }
        }
    }
    return ret;
}