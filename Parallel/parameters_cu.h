/* parameters.h declares common parameters. */

#ifndef PARAMETERS_CU_H
#define PARAMETERS_CU_H

#define THREADS_PER_BLOCK 4

#define Nx 100
#define Ny Nx

// Boundary type parameters
// 1: "linear", 2: "sinunoidal", default: "constant"
#define BOUNDARY 1

// Error tolerence parameter
#define ERROR_DECIMAL_DIG 2
#define ERROR 1e-2
#define NLOOPS 10

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
