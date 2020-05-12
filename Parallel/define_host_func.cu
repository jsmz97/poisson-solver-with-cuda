#ifndef DEFINE_HOST_FUNC_CU
#define DEFINE_HOST_FUNC_CU

#include <stdlib.h>    
#include <stdio.h>
#include "parameters_cu.h"

/* Write Results */
__host__ void output (FILE *file, float grid_value[Nx][Ny]){
    int i, j;

    for(i = 0; i < Nx; i++){
        for(j = 0; j < Ny; j++){
            fprintf(file, "%.*f\n", ERROR_DECIMAL_DIG, grid_value[i][j]);
        }
        fprintf(file, "\n");
    }
}

/* Save Results */
__host__ void save_results(float grid_value[Nx][Ny]){
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
