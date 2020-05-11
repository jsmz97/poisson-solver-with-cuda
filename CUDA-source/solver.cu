// #include <stdio.h>   // included in aux_func.c for debug
#ifndef KERNEL_CU
#define KERNEL_CU

#include <stdio.h>
#include "declare_func.h"
#include "define_global_func.cu"
#include "define_host_func.cu"
#include "parameters.h"
#include "grid.h"

// __device__ int *d_p_not_tolerent;

__host__ void solve_laplace(Grid_t *h_u){
    Grid_t *d_u;
    size_t size_matrix = sizeof(h_u->value);

    cudaMalloc(&d_u, sizeof(Grid_t));

    typeof(d_not_tolerent) h_not_tolerent;

    // Use this counter to do tolerence_test every NLOOPS loops.
    int counter = 0;

    // Invoke kernel.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((Nx + dimBlock.x - 1) / dimBlock.x,
                 (Ny + dimBlock.y - 1) / dimBlock.y);
    
    // Initialize grid values, and set boundary conditions.
    grid_init<<<dimGrid, dimBlock>>>(d_u);

    // None of the inner part of the grid are used
    // when calling boundary().
    boundary<<<dimGrid, dimBlock>>>(d_u, BOUNDARY);

    // The main loop to calculate the function.
    do{
        // update_old_grid<<<dimGrid, dimBlock>>>(d_u);
        calc_grid<<<dimGrid, dimBlock>>>(d_u);

        // Run tolerence_test every NLOOPS loops.
        if(!counter){
            // Following https://stackoverflow.com/a/2637310
            reset_d_not_tolerent<<<1, 1>>>();
            tolerence_test<<<dimGrid, dimBlock>>>(d_u, ERROR);
            cudaMemcpyFromSymbol(&h_not_tolerent, d_not_tolerent, sizeof(d_not_tolerent));
        }
        counter = (counter + 1) % NLOOPS;
    } while(h_not_tolerent);

    // Copy memory from device to host.
    cudaMemcpy(h_u->value, d_u->value, size_matrix,
                cudaMemcpyDeviceToHost);

     // Free device memory
     cudaFree(d_u);
}

#endif
