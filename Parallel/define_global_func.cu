/* define_global_functions defines functions that are executed on the device */

#ifndef DEFINE_GLOBAL_FUNC_CU
#define DEFINE_GLOBAL_FUNC_CU

#include <stdio.h>
#include "parameters_cu.h"
#include "grid_cu.h"

/* Grid Initialization */
__global__ void grid_init (Grid_t *u){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < Nx && j < Ny){
        u->value[i][j] = 0;
    }
}


/* Boundary Value Initialization */
__global__ void boundary_init (Grid_t *u, int type){
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


/* Calculation of Node Values Using Jacobi Iteration */
__global__ void grid_update (Grid_t *u){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    typeof(u->value[0][0]) value_new;

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


/* Convergent Test */

// Define a boolean value for loop control.
__device__ unsigned int d_not_tolerent;

// Define a value that resets d_not_tolerent.
__global__ void reset_d_not_tolerent (){
    d_not_tolerent = 0;
    // 0 = true ; else = false.
}


// Define a function that carry out the convergence test.
__global__ void is_convergent (Grid_t *u, float tolerence){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1){
        if(fabsf(u->residual[i][j]) > fabsf(tolerence)){
            // If the convergence criterium is not satisfied, return false.
	    // A simple way is to increment d_not_tolerent.
            d_not_tolerent++;
        }
    }
}

#endif
