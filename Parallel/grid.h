// struct_cu.h defines structures used by other source files.

#ifndef STRUCT_VAR_CU_H
#define STRUCT_VAR_CU_H

#include "parameters.h"     // For access to Nx, Ny

// Following https://stackoverflow.com/a/8095711
typedef struct  // use typedef for convenience, no need to type "struct" all over the place
{
    // float value_new[Nx][Ny];
    // float value_old[Nx][Ny];

    float value[Nx][Ny];
    // Residuals on the oundary are useless, but kept for convenient.
    float residual[Nx][Ny];
} Grid_t;

__device__ unsigned int d_not_tolerent;

#endif
