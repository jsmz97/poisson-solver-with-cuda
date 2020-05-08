#ifndef HEAD_H
#define HEAD_H

#include <stdio.h>
#include "parameters.h"
#include "struct.h"

// The following functions are all defined in aux_func.c

void grid_init (Grid_t *u);

void boundary (Grid_t *u, int type);

// Successive Overrelaxation function.
float sor (float value_old, float residual);
// float sor (float value_new, float value_old, float residual);

void calc_grid (Grid_t *u);
// void calc_grid_no_GS (Grid_t *u);


// void update_new_grid (Grid_t *u);

// void update_old_grid (Grid_t *u);

_Bool is_tolerent (float residual[Nx][Ny], float tolerence);

void output (FILE *file, float grid_value[Nx][Ny]);

_Bool is_exceed (float value, float max);

// Save the results to the ouput file.
void save_results(float grid_value[Nx][Ny]);

#endif