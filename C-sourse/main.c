// #include <stdio.h>   // included in aux_func.c for debug
#include "head.h"
#include "parameters.h"
#include "aux_func.c"

// // Error tolerence
// #define ERROR_DECIMAL_DIG 2
// #define ERROR 1e-ERROR_DECIMAL_DIG

// // Boundary type
// // 1: "constant", 2: "sinunoidal"
// #define BOUNDARY 2

int main(){
    Grid_t u;

    // Following https://stackoverflow.com/a/51009276
    u = (Grid_t)malloc(sizeof(*u));

    // Initialize grid values, and set boundary conditions.
    grid_init(&u);
    boundary(&u, BOUNDARY);

    // Use this counter to do tolerence_test every NLOOPS loops.
    int counter = 0;
    int tolerent;

    // The main loop to calculate the function.
    do{
        calc_grid(&u);

        if(!counter){
            tolerent = is_tolerent(u->residual, ERROR);
        }
        counter = (counter + 1) % NLOOPS;
    } while (!tolerent);
    // while(is_exceed(max_residual(u->residual), ERROR));


    // Save results to a file.
    save_results(u->element);

    // Free the pointer.
    free(u);

    return 0;
}