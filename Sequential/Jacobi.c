#include <stdio.h> 
#include "declare_func.h"
#include "parameters.h"
#include "define_func.c"

int main(){

    /* Allocate memory for the grid. */
    Grid_t grid;
    grid = (Grid_t)malloc(sizeof(*grid));

    /* Initialize grid values and boundary conditions. */
    grid_init(&grid);
    boundary_init(&grid, BOUNDARY);

    int convergent;

    /* Call main Loop for calculation */
    do{
        grid_update(&grid, SOURCE);
        convergent = is_convergent(grid->residual, ERROR);
    } while (!convergent);

    /* Save results to a file */
    save_results(grid->value);

    /* Free the memory & pointer */
    free(grid);
    grid = NULL;

    return 0;
}