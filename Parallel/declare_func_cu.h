#ifndef DECLARE_FUNC_CU_H
#define DECLARE_FUNC_CU_H

#include "grid_cu.h"

__global__ void grid_init (Grid_t *u);

__global__ void boundary (Grid_t *u, int b_type);

__global__ void calc_grid (Grid_t *u);

__global__ void is_convergent (Grid_t *u, float tolerence);

__global__ void reset_d_not_tolerent ();

__device__ int is_exceed (float value, float max);

__host__ void solve_laplace(Grid_t *h_u);

__host__ void output (FILE *file, float grid_value[Nx][Ny]);

__host__ void save_results(float grid_value[Nx][Ny]);

#endif
