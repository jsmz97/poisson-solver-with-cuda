// aux_func.c provide functions for boundary conditions, finite difference function for Laplace equation, etc.

#ifndef AUX_FUNC_C
#define AUX_FUNC_C

#include <stdlib.h>     // For exit(1)
#include <stdio.h>
#include <string.h>     // For memcpy()
// #include <stdlib.h>     // For abs(int) (for integers only!)
#include <math.h>       // For fabsf(float), M_PI, sinf(float) (need -lm for sinf() on compilation).

#include "head.h"
#include "struct.h"


// Initial the grid with boundary conditions. The pointer coordinate points to a 2D array temperarily.
void grid_init (Grid_t *u){
    // float ** for 2D array. Following https://stackoverflow.com/a/19908521
    int i, j;
    // Initial all grid values to be 0.
    for(i = 0; i < Nx; i++)
        for(j = 0; j < Ny; j++){
            (*u)->element[i][j] = 0.;
        }
}

// Boundary conditions.
// Note: Boundary conditions should always be set to both element and value_old.
void boundary (Grid_t *u, int b_type){
    int i;
    switch (b_type)
    {
    case 1:
        // Linear initial boundary values.
        for(i = 0; i < Ny; i++){
            (*u)->element[0][i] = i * 100. / Ny;
            (*u)->element[i][Ny-1] = (Ny - 1 - i) * 100. / Nx;
            (*u)->element[Nx-1][i] = (i - Ny + 1) * 100. / Ny;
            (*u)->element[i][0] = (-i) * 100. / Nx;
        }
        break;
    case 2:
        // Sinunoidal initial boundary values.
        for(i = 0; i < Ny; i++){
            // NOTE: sinf() requires parameter '-lm' when compiling.
            (*u)->element[0][i] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);            
            (*u)->element[i][Ny-1] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);            
            (*u)->element[Nx-1][i] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
            (*u)->element[i][0] = 100. * sinf( ((float) i) / Ny * 2 * M_PI);
        }
        break;
    default:
        // Constant initial boundary values.
        // For x = 0, grid_value's are 100; otherwise 0.
        for(i = 0; i < Nx; i++){
            (*u)->element[0][i] = 100.;
        }
        break;
    }
}


// Successive Overrelaxation function.
float sor (float value_old, float residual){
    return value_old + OMEGA * residual;
    // return value_new;
}



// // Calculate the new grid values, using Gauss-Seidel method, see eq. (17.28) in Landau.pdf
// void calc_grid (Grid_t *u){
//     int i, j;
    
//     // The following calculation supposes a rectanglular region.
//     for(i = 1; i < Nx - 1; i++)
//         for(j = 1; j < Ny - 1; j++){
//             // Temperary new grid value by Gauss-Seidel method.
//             (*u)->value_new[i][j] = 0.25 * ((*u)->value_old[i+1][j] + (*u)->value_old[i][j+1] +   // old ones
//                                          (*u)->value_new[i-1][j] + (*u)->value_new[i][j-1]);   // new ones
//             (*u)->residual[i][j] = (*u)->value_new[i][j] - (*u)->value_old[i][j];
            
//             // The exact new grid value by SOR technique.
//             (*u)->value_new[i][j] = sor((*u)->value_old[i][j], (*u)->residual[i][j]);
//             // printf("new grid value for u[%d][%d]: %.1f\n", i, j, (*u)->value_new[i][j]);
//         }
// }


// Calculate the new grid values, USING Gauss-Seidel method, see eq. (17.28) in Landau.pdf
void calc_grid (Grid_t *u){
    int i, j;
    typeof((*u)->element[0][0]) element_new;
    // The following calculation supposes a rectanglular region.
    for(i = 1; i < Nx - 1; i++)
        for(j = 1; j < Ny - 1; j++){
            // Temperary new grid value by Gauss-Seidel method.
            element_new = 0.25 * ((*u)->element[i+1][j] + (*u)->element[i][j+1] +
                          (*u)->element[i-1][j] + (*u)->element[i][j-1]);
            (*u)->residual[i][j] = element_new - (*u)->element[i][j];
            // (*u)->element[i][j] = element_new;

            // // Test if there is any difference by merging the new and old arrays.
            // // Temperary new grid value by Gauss-Seidel method.
            // (*u)->value_new[i][j] = 0.25 * ((*u)->value_old[i+1][j] + (*u)->value_old[i][j+1] +   // old ones
            //                              (*u)->value_old[i-1][j] + (*u)->value_old[i][j-1]);   // new ones
            // (*u)->residual[i][j] = (*u)->value_new[i][j] - (*u)->value_old[i][j];
            
            // The exact new grid value by SOR technique.
            (*u)->element[i][j] = sor((*u)->element[i][j], (*u)->residual[i][j]);
        }
}


// // Update the old grid values.
// void update_old_grid (Grid_t *u){
//     // Memmory copy from new to old, using memcpy.
//     memcpy((*u)->value_old, (*u)->value_new, sizeof((*u)->value_new));

//     // int i, j;
//     // for(i = 1; i < Nx - 1; i++)
//     //     for(j = 1; j < Ny - 1; j++){
//     //         // The previously"new" grid values become now "old" values in a new round.
//     //         (*u)->value_old[i][j] = (*u)->value_new[i][j];
//     //     }
// }


// // Return the maximum of the absolute value of residuals.
// float max_residual (float residual[Nx][Ny]){
//     int i, j;
//     float v_ij;
//     // float max = fabsf(residual[1][1]);
//     float max = residual[1][1];

//     // printf("Initial of max is %.1f\n", max);
    
//     for(i = 1; i < Nx - 1; i++){
//         for(j = 1; j < Ny - 1; j++){
//             v_ij = residual[i][j];
//             // v_ij = fabsf(residual[i][j]);

//             // printf("r[%d][%d] = %5.1f, r[%d][%d] = %5.1f\n", i, j, residual[i][j], i, j, v_ij);
//             // if(is_exceed(v_ij, 100.)){
//             //     printf("Error: r[%d][%d] = %.1f exceeds the maximum!\n", i, j, residual[i][j]);
//             // }
//             max = is_exceed(max, v_ij) ? max : v_ij;
//         }
//     }
    
//     // printf("max of absolute residual is %.1f\n", max);
//     // if(is_exceed(max, 100.)){
//     //     printf("Error: max of absolute residual exceeds the maximum\n");
//     // }
//     return fabsf(max);
// }


// Test if the residuals are all smaller than the error tolerence.
_Bool is_tolerent (float residual[Nx][Ny], float tolerence){
    int i, j;
    for(i = 1; i < Nx - 1; i++){
        for(j = 1; j < Ny - 1; j++){
            if(is_exceed(residual[i][j], tolerence))
                return 0;
        }
    }
    return 1;
}


// Write the results to a file with the given file name.
void output (FILE *file, float grid_value[Nx][Ny]){
    int i, j;
    // Print the title
    // fprintf(file, "# x,\ty,\tvalue\n");
    for(i = 0; i < Nx; i++){
        for(j = 0; j < Ny; j++){
            fprintf(file, "%.*f\n", ERROR_DECIMAL_DIG, grid_value[i][j]);
        }
        fprintf(file, "\n");
    }
}


// Test if the value exceeds the maximul value. If so, print error info.
_Bool is_exceed (float value, float max){
    // if(fabsf(value) > fabsf(max)){
    //     // printf("Error: value exceeds the maximum!\n");
    //     return 1;
    // }
    // else
    //     return 0;
    return (fabsf(value) > fabsf(max));
}


// // Return a matrix element
// // Thanks Glenn G.
// // Following https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// float GetElement(const Grid_t *u, int x, int y){
//     return u.element[x * Ny + y];
// }


// Save the results to the ouput file.
void save_results(float grid_value[Nx][Ny]){
    FILE *file;

    file = fopen(FILENAME, "w");      // Following https://stackoverflow.com/a/9840678

    if(file == NULL)
    {
        printf("FILE pointer NULL!");   
        exit(1);             
    }

    output(file, grid_value);
    fclose(file);
}

#endif
