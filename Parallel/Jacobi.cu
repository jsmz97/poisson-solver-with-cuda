#include "define_host_func.cu"
#include "solver.cu"
#include "grid_cu.h"

int main(){
    /* Allocate memory for grid on host. */
    Grid_t *h_u = (Grid_t*)malloc(sizeof(Grid_t));

    /* Call the solver to invoke our kernels. */
    solver(h_u);

    /* Save results to the output file. */
    save_results(h_u->value);

    /* Free the host memory & pointer. */
    free(h_u);
    h_u = NULL;

    return 0;
}
