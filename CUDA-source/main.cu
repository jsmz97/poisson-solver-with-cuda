// #include <stdio.h>   // included in aux_func.c for debug
// #include <stdlib.h>     // For exit(1)

#include "host_device_func.cu"
#include "kernel.cu"
#include "struct_var_cu.h"
// #include "parameters_cu.h"

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