/* func.c defines functions that can be called in main.c. */

#ifndef DEFINE_FUNC_C
#define DEFINE_FUNC_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h> 
#include <math.h> 

#include "declare_func.h"
#include "grid.h"

/* Square Tiling Using SoA */
// Note: We initialize the pointer to the SoA, instead of the SoA itself.

/* Grid Initialization */
void grid_init (Grid_t *u){
    // Initial all grid values to be 0.
    for(int i = 0; i < Nx; i++)
        for(int j = 0; j < Ny; j++){
            (*u)->value[i][j] = 0.;
        }
}

/* Boundary Value Initialization */
void boundary_init (Grid_t *u, int type){
    switch (type)
    {
    case 1:
        // Linear BVC.
        for(int i = 0; i < Ny; i++){
            (*u)->value[0][i] = i * 100. / Ny;
            (*u)->value[i][Ny-1] = (Ny - 1 - i) * 100. / Nx;
            (*u)->value[Nx-1][i] = (i - Ny + 1) * 100. / Ny;
            (*u)->value[i][0] = (-i) * 100. / Nx;
        }
        break;
    case 2:
        // Sinunoidal BVC.
        for(int i = 0; i < Ny; i++){
            // NOTE: sinf() requires parameter '-lm' when compiling.
            (*u)->value[0][i] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);            
            (*u)->value[i][Ny-1] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);            
            (*u)->value[Nx-1][i] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
            (*u)->value[i][0] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
        }
        break;
    case 3:
        // Constant BVC
        for(int i = 0; i < Nx; i++){
            (*u)->value[0][i] = 100.;
            (*u)->value[Nx-1][i] = -100.;
        }
        break;
    default:
        // Zero Constant BVC
        break;
    }
}

/* Calculation of Node Values Using Jacobi Iteration */
void grid_update (Grid_t *u, int type){
    // Declare the intermediate result
    typeof((*u)->value[0][0]) value_new;
    // Declare the source term
    //typeof((*u)->value) source;
    // Initialize the source term
    // for(int i = 0; i < Nx; i++)
    //    for(int j = 0; j < Ny; j++){
    //        source[i][j] = 0.;
    //}
    //int x_mid = Nx/2;
    //int y_mid = Nx/2;
    // Add source term or not
    //switch(type){
        // Pseudo-delta
        //case 1: source[x_mid][y_mid] = 100; break;
        // 
        //default: break;
    //}

    // Iteration over the grid except for i or j = 0 or N-2
    for(int i = 1; i < Nx - 1; i++)
        for(int j = 1; j < Ny - 1; j++){
            // Calculate the new value at the current node
            value_new = 0.25 * ((*u)->value[i+1][j] + (*u)->value[i][j+1] +
                          (*u)->value[i-1][j] + (*u)->value[i][j-1]);
            // Save the residual for convergence test
	        (*u)->residual[i][j] = value_new - (*u)->value[i][j];
            // Update the values of the current node
            (*u)->value[i][j] = value_new;
        }
}

/* Convergence Test */
// Note: Convergence is ONLY achieved if residuals at ALL nodes are smaller than the tolerance.
_Bool is_convergent (float residual[Nx][Ny], float tolerence){
    // By definition, convergence is achieved if residuals at all nodes are tolerated
    for(int i = 1; i < Nx - 1; i++){
        for(int j = 1; j < Ny - 1; j++){
            if(fabsf(residual[i][j]) >= fabsf(tolerence))
                return 0;
        }
    }
    return 1;
}

/* Write Results */
void output (FILE *file, float grid_value[Nx][Ny]){
    for(int i = 0; i < Nx; i++){
        for(int j = 0; j < Ny; j++){
            fprintf(file, "%.*f\n", ERROR_DECIMAL_DIG, grid_value[i][j]);
        }
        fprintf(file, "\n");
    }
}

/* Save Results */
void save_results(float grid_value[Nx][Ny]){
    FILE *file;

    file = fopen(FILENAME, "w");

    if(file == NULL)
    {
        printf("FILE pointer NULL!");   
        exit(1);             
    }

    output(file, grid_value);
    fclose(file);
}

#endif
