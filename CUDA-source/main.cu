// #include <stdio.h>   // included in aux_func.c for debug
// #include <stdlib.h>     // For exit(1)

#include "define_host_func.cu"
#include "solver.cu"
#include "grid.h"


int main(){
    // Grid_t h_u;
    Grid_t *h_u = (Grid_t*)malloc(sizeof(Grid_t));

    // Solve the Laplace's equation and save results to h_u->element.
    solve_laplace(h_u);

    // Save results to a file.
    save_results(h_u->element);

    // Free the pointer.
    free(h_u);

    return 0;
}
