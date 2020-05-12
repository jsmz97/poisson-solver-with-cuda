/* grid_cu.h defines the 2-D computational grid as SoA. */

#ifndef GRID_H
#define GRID_H

#include "parameters_cu.h"     // For access to Nx, Ny

// Following https://stackoverflow.com/a/8095711
typedef struct  // use typedef for convenience, no need to type "struct" all over the place
{
    // float value_new[Nx][Ny];
    // float value_old[Nx][Ny];

    float value[Nx][Ny];
    // Residuals on the oundary are useless, but kept for convenient.
    float residual[Nx][Ny];
} Grid_t;

#endif
