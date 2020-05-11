// aux_func.cu implement functions in aux_func.c to kernel functions.
#ifndef GLOBAL_FUNC_CU
#define GLOBAL_FUNC_CU

#include <stdio.h>
#include "parameters.h"
#include "grid.h"

// Kernel definition with __global__
__global__ void grid_init (Grid_t *u){
    // blockIdx, blockDim, threadIdx are built-in kernel variables.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < Nx && j < Ny){   // All range
        u->element[i][j] = 0;
    }
}


// Boundary conditions.
__global__ void boundary (Grid_t *u, int b_type){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < Nx && j < Ny){
    switch(b_type){
    case 1:    // Linear initial boundary values.
        if(i == 0){
            // grid[0][j].
            u->element[0][j] = j * 100. / Ny;
        }
        else if(i == Nx - 1){
            // grid[Nx-1][j].
            u->element[Nx-1][j] = (j - Ny + 1) * 100. / Ny;
        }
        else if(j == 0){
            // grid[i][0].
            u->element[i][0] = (-i) * 100. / Nx;
        }
        else if(j == Ny - 1){
            // grid[i][Ny-1].
            u->element[i][Ny-1] = (Nx - 1 - i) * 100. / Nx;
        }
        break;
    case 2:    // Sinunoidal initial boundary values.
        if(i == 0){
            // grid[0][j].
            u->element[0][j] = 100. * sinf( ((float) j) / Ny * 2 * M_PI);
        }
        else if(i == Nx - 1){
            // grid[Nx-1][j].
            u->element[Nx-1][j] = 100. * sinf( ((float) j) / Ny * 2 * M_PI);
        }
        if(j == 0){
            // grid[i][0].
            u->element[i][0] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
        }
        else if(j == Ny - 1){
            // grid[i][Ny-1].
            u->element[i][Ny-1] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
        }
        break;
    case 3:     // Linear, discontinuous at one end point.
        if(i == 0){
            // grid[0][j].
            u->element[0][j] = (float)j * 25. / Ny;
        }
        else if(i == Nx - 1){
            // grid[Nx-1][j].
            u->element[Nx-1][j] = (3. * Ny - 3.0 - j) * 25. / Ny;
        }
        else if(j == 0){
            // grid[i][0].
            u->element[i][0] = (4. * Nx - 4.0 - i) * 25. / Nx;
        }
        else if(j == Ny - 1){
            // grid[i][Ny-1].
            u->element[i][Ny-1] = (Ny - 1. + i) * 25. / Nx;
        }
        break;
    case 4:	// Linear, four discontinuous end points
		if(i == 0){
            // grid[0][j].
	    	if(j < Ny / 2)
            	u->element[0][j] = j * 100. / Ny;
        	else
        		u->element[0][j] = (j - Ny + 1) * 100. / Ny;
        }
        else if(i == Nx - 1){
            // grid[Nx-1][j].
            if(j < Ny / 2)
	            u->element[Nx-1][j] = -j * 100. / Ny;
            else
            	u->element[Nx-1][j] = (Ny - 1 - j) * 100. / Ny;
        }
        else if(j == 0){
            // grid[i][0].
            if(i < Nx / 2)
	            u->element[i][0] = -i * 100. / Nx;
            else
            	u->element[i][0] = (Nx - 1 - i) * 100. / Nx;
        }
        else if(j == Ny - 1){
            // grid[i][Ny-1].
            if(i < Nx / 2)
            	u->element[i][Ny-1] = i * 100 / Nx;
        	else
        		u->element[i][Ny-1] = (i - Nx - 1) * 100. / Nx;
        }
        break;
    default:    // Constant initial boundary values.
        // For x = 0, grid_value's are 100; otherwise 0.
        if(i == 0){
            u->element[0][j] = 100.;
        }
        break;
    }
    }
}


// Calculate the new function values, USING Gauss-Seidel method.
__global__ void calc_grid (Grid_t *u){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    typeof(u->element[0][0]) element_new;

    // The following calculation supposes a rectanglular region.
    if(i < Nx - 1 && i > 0 && j < Ny - 1 && j > 0){   // Inner part
        // // Temperary new grid value without Gauss-Seidel method.
        // element_new = 0.25 * (u->value_old[i+1][j] + u->value_old[i][j+1] +
        //                       u->value_old[i-1][j] + u->value_old[i][j-1]);
        
        // Temperary new grid value using Gauss-Seidel method.
        element_new = 0.25 * (u->element[i+1][j] + u->element[i][j+1] +
                              u->element[i-1][j] + u->element[i][j-1]);

        // // Synchronize threads to ensure value_new[][] are new.
        // __syncthreads();

        u->residual[i][j] = element_new - u->element[i][j];

        // u->value_new[i][j] = element_new;
        // // Synchronize threads to ensure value_new[][] are new.
        // __syncthreads();

        // Successive Overrelaxation (SOR) method
        u->element[i][j] = element_new;
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
