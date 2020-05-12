/* parameters.h declares common parameters. */

#ifndef PARAMETER_H
#define PARAMETER_H

// Matrix order parameters
#define Nx 1000
#define Ny Nx

// Error tolerence paramters
#define ERROR_DECIMAL_DIG 2
#define ERROR 1e-2

// Boundary type parameter
// Note: 1: linear, 2: sinunoidal, 3: constant default: zeroc constant
#define BOUNDARY 1

// Source term parameter
// Note: 1: delta,  default: sourceless
#define SOURCE 0



// Output parameters
#define DIRNAME "Outputs"
#define SUFFIX ".dat"
#define STR_HELPER(INT) #INT
#define STR(INT) STR_HELPER(INT)
#define CONCATENATE(DIR, SOU, BD, SZ, DIG, SFX) DIRNAME "/" "Solution_SOURCE_" STR(SOU) "_BOUNDARY_" STR(BD) "_SIZE_" STR(SZ) "_SIGMA_" STR(DIG) SFX
#define FILENAME CONCATENATE(DIRNAME, SOURCE, BOUNDARY, Nx, ERROR_DECIMAL_DIG, SUFFIX)

#endif
