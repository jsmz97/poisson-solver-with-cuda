// parameters_cu.h decfines macros only used by functions in .cu
#ifndef PARAMETERS_CU_H
#define PARAMETERS_CU_H

// A thread block size of 16x16 (256 threads),
// although arbitrary in this case, is a common choice.
// The grid is created with enough blocks to have
// one thread per matrix element as before.
#define BLOCK_SIZE 16

#define Nx 1000
#define Ny Nx

// Boundary type
// 1: "linear", 2: "sinunoidal", 3: "linear, one discontinuous point", 4: "linear, four discontinuous points",
// any others: "constant"
#define BOUNDARY 0 

// Error tolerence
#define ERROR_DECIMAL_DIG 1
#define ERROR 1e-1

// OMEGA in SOR method.
#define OMEGA 1.0

#define DIRNAME "Datafile"
#define SUFFIX "_use_GS.csv"

// Following https://stackoverflow.com/a/5459929
// Concatenate integers with strings.
#define STR_HELPER(INT) #INT
#define STR(INT) STR_HELPER(INT)

#define CONCATENATE(DIR, BD, SZ, DIG, OMG, SFX) DIRNAME "/" "boundary_" STR(BD) "_size_" STR(SZ) "_dig_" STR(DIG) "_omega_" STR(OMG) SFX
#define FILENAME CONCATENATE(DIRNAME, BOUNDARY, Nx, ERROR_DECIMAL_DIG, OMG, SUFFIX)

// Thanks for Stefan Recksiegel to point it out that I can run tolerence_test every ten loops.
#define NLOOPS 1

#endif
