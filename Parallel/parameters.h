/* parameters.h declares common parameters. */

#ifndef PARAMETERS_CU_H
#define PARAMETERS_CU_H

// A thread block size of 16x16 (256 threads),
// although arbitrary in this case, is a common choice.
// The grid is created with enough blocks to have
// one thread per matrix element as before.
#define BLOCK_SIZE 16

#define Nx 1000
#define Ny Nx

// Boundary type parameters
// 1: "linear", 2: "sinunoidal", 3: "linear, one discontinuous point", 4: "linear, four discontinuous points",
// any others: "constant"
#define BOUNDARY 0 

// Error tolerence parameter
#define ERROR_DECIMAL_DIG 1
#define ERROR 1e-1
#define NLOOPS 1

// Source term paramter
#define SOURCE 0

// Output parameters
#define DIRNAME "Outputs"
#define SUFFIX ".dat"
#define STR_HELPER(INT) #INT
#define STR(INT) STR_HELPER(INT)
#define CONCATENATE(DIR, SOU, BD, SZ, DIG, SFX) DIRNAME "/" "Solution_SOURCE_" STR(SOU) "_BOUNDARY_" STR(BD) "_SIZE_" STR(SZ) "_SIGMA_" STR(DIG) SFX
#define FILENAME CONCATENATE(DIRNAME, SOURCE, BOUNDARY, Nx, ERROR_DECIMAL_DIG, SUFFIX)

#endif
