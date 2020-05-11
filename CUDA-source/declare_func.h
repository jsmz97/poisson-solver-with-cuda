// head_cu.h declares all the functions used in kernel.cu

#ifndef HEAD_CU_H
#define HEAD_CU_H

#include "grid.h"

// Kernel definition with __global__
__global__ void grid_init (Grid_t *u);

// Boundary conditions.
__global__ void boundary (Grid_t *u, int b_type);

// Calculate the new function values, NOT using Gauss-Seidel method.
__global__ void calc_grid (Grid_t *u);

// Update the old grid values.
// __global__ void update_old_grid (Grid_t *u);

// Test whether residuals are within tolerence.
__global__ void tolerence_test (Grid_t *u, float tolerence);

// Reset the *d_p_not_tolerent to 0.
__global__ void reset_d_not_tolerent ();

// Test if the value exceeds the maximul value. If so, print error info.
__device__ int is_exceed (float value, float max);

// The function to solve the Laplace equation.
__host__ void solve_laplace(Grid_t *h_u);

// fprintf() undefined in device code
__host__ void output (FILE *file, float grid_value[Nx][Ny]);

// Process save of the results.
__host__ void save_results(float grid_value[Nx][Ny]);

#endif
