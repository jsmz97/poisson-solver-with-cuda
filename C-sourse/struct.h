// struct.h defines structures used by other source files.

#ifndef STRUCT_H
#define STRUCT_H

#include "parameters.h"     // For access to Nx, Ny

// Previously following https://stackoverflow.com/a/8095711
// Now following https://stackoverflow.com/q/1543713
typedef struct // use typedef for convenience, no need to type "struct" all over the place
{
    // float value_new[Nx][Ny];
    // float value_old[Nx][Ny];
    float element[Nx][Ny];
    // Residuals on the oundary are useless, but kept for convenient.
    float residual[Nx][Ny];
} *Grid_t;
// or even
// Grid_t, *pGrid_t;

#endif