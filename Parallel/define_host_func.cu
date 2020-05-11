// aux_func.cu implement functions in aux_func.c to kernel functions.
#ifndef HOST_DEVICE_FUNC_CU
#define HOST_DEVICE_FUNC_CU

#include <stdlib.h>     // For exit(1)
#include <stdio.h>
#include "parameters.h"

// Test if the value exceeds the maximul value. If so, print error info.
__device__ int is_exceed (float value, float max){
    return (fabsf(value) > fabsf(max));
}


// Write the results to a file with the given file name.
__host__ void output (FILE *file, float grid_value[Nx][Ny]){
    int i, j;
    // Output for gnuplot splot.
    for(i = 0; i < Nx; i++){
        for(j = 0; j < Ny; j++){
            fprintf(file, "%.*f\n", ERROR_DECIMAL_DIG, grid_value[i][j]);
        }
        fprintf(file, "\n");
    }
}


// Process save of the results.
__host__ void save_results(float grid_value[Nx][Ny]){
    FILE *file;

    // Save the results to the ouput file.
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
