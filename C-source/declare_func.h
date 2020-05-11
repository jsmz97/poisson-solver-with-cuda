/* delcare_func.h declares functions defined in define_func.c */
#ifndef DECLARE_FUNC_H
#define DECLARE_FUNC_H

#include <stdio.h>
#include "parameters.h"
#include "grid.h"

// The following functions are all defined in aux_func.c

void grid_init (Grid_t *grid);

void boundary_init (Grid_t *grid, int type);

void grid_update (Grid_t *grid, int source_type);

_Bool is_tolerent (float residual[Nx][Ny], float tolerence);

void output (FILE *file, float grid_value[Nx][Ny]);

void save_results(float grid_value[Nx][Ny]);

#endif