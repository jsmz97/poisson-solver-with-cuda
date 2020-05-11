// aux_func.cu implement functions in aux_func.c to kernel functions.
#ifndef GLOBAL_FUNC_CU
#define GLOBAL_FUNC_CU

#include <stdio.h>
#include "parameters.h"
#include "grid.h"

/* Grid Initialization */
__global__ void grid_init (Grid_t *u){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < Nx && j < Ny){
        u->value[i][j] = 0;
    }
}


/* Boundary Value Initialization */
__global__ void boundary (Grid_t *u, int type){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < Nx && j < Ny){
        switch(type){
        case 1:    // Linear BVC
            if(i == 0){
                // grid[0][j].
                u->value[0][j] = j * 100. / Ny;
            }
            else if(i == Nx - 1){
                u->value[Nx-1][j] = (j - Ny + 1) * 100. / Ny;
            }
            else if(j == 0){
                u->value[i][0] = (-i) * 100. / Nx;
            }
            else if(j == Ny - 1){
                u->value[i][Ny-1] = (Nx - 1 - i) * 100. / Nx;
            }
            break;
        case 2:    // Sinunoidal BVC
            if(i == 0){
                u->value[0][j] = 100. * sinf( ((float) j) / Ny * 2 * M_PI);
            }
            else if(i == Nx - 1){
                u->value[Nx-1][j] = 100. * sinf( ((float) j) / Ny * 2 * M_PI);
            }
            if(j == 0){
                u->value[i][0] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
            }
            else if(j == Ny - 1){
                u->value[i][Ny-1] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
            }
            break;

        default:    // Constant BVC
            if(i == 0){
                u->value[0][j] = 100.;
            }
            break;
        }
    }
}


// Calculate the new function values, USING Gauss-Seidel method.
__global__ void calc_grid (Grid_t *u){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    typeof(u->value[0][0]) value_new;

    // The following calculation supposes a rectanglular region.
    if(i < Nx - 1 && i > 0 && j < Ny - 1 && j > 0){

        // Calculate the new value at the current node
        value_new = 0.25 * (u->value[i+1][j] + u->value[i][j+1] +
                              u->value[i-1][j] + u->value[i][j-1]);
        // Save the residual for convergence test
        u->residual[i][j] = value_new - u->value[i][j];
        // Update the values of the current node
        u->value[i][j] = value_new;
    }
}


// Following https://stackoverflow.com/a/2637310
// Reset the *d_p_not_tolerent to 0.
__global__ void reset_d_not_tolerent (){
    d_not_tolerent = 0;
}


// Test whether residuals are within tolerence.
__global__ void tolerence_test (Grid_t *u, float tolerence){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1){
        if(is_exceed(u->residual[i][j], tolerence)){
            // d_not_tolerent works as a bool type value.
            d_not_tolerent++;
        }
    }
}

#endif
