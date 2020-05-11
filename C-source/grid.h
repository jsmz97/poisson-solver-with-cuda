/* grid.h defines the 2-D computational grid as SoA. */

#ifndef GRID_H
#define GRID_H

#include "parameters.h"

// Square Tiling Using SoA

typedef struct
{
    float value[Nx][Ny];
    // Array to store the values on nodes
    float residual[Nx][Ny];
    // Array to store the residuals on nodes
} *Grid_t;

#endif