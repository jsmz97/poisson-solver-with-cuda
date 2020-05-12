/* solver.cu defines the solver function that invokes kernel functions */

#ifndef SOLVER_CU
#define SOLVER_CU

#include <stdio.h>
#include "declare_func_cu.h"
#include "define_global_func.cu"
#include "define_host_func.cu"
#include "parameters_cu.h"
#include "grid_cu.h"

__host__ void solver(Grid_t *h_u){
    Grid_t *d_u;
    size_t size_matrix = sizeof(h_u->value);


    cudaMalloc(&d_u, sizeof(Grid_t));

    typeof(d_not_tolerent) h_not_tolerent;

    /* Intialize the grid & block sizes */
    dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 dimGrid((Nx + dimBlock.x - 1) / dimBlock.x,
                 (Ny + dimBlock.y - 1) / dimBlock.y);
    
    /* Initialize the grid */
    grid_init<<<dimGrid, dimBlock>>>(d_u);

    /* Set the boundary value */
    boundary_init<<<dimGrid, dimBlock>>>(d_u, BOUNDARY);

    /*Call main Loop for calculation
    do{  
	grid_update<<<dimGrid, dimBlock>>>(d_u);    
        reset_d_not_tolerent<<<1, 1>>>();
        is_convergent<<<dimGrid, dimBlock>>>(d_u, ERROR);
        cudaMemcpyFromSymbol(&h_not_tolerent, d_not_tolerent, sizeof(d_not_tolerent));

    } while(h_not_tolerent);*/

    int counter = 0;

    do{
        grid_update<<<dimGrid, dimBlock>>>(d_u);

        if(!counter){
            
            reset_d_not_tolerent<<<1, 1>>>();
            is_convergent<<<dimGrid, dimBlock>>>(d_u, ERROR);
            cudaMemcpyFromSymbol(&h_not_tolerent, d_not_tolerent, sizeof(d_not_tolerent));
        }
        counter = (counter + 1) % NLOOPS;
    } while(h_not_tolerent);


    /* Copy memory from device to host. */
    cudaMemcpy(h_u->value, d_u->value, size_matrix, cudaMemcpyDeviceToHost);

    /* Free device memory & pointer. */
    cudaFree(d_u);
    d_u = NULL;
}

#endif
