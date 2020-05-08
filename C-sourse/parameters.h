// parameters.h declares common parameters.

#ifndef PARAMETER_H
#define PARAMETER_H

// size of each side of the square/rectangle
#define Nx 5000
#define Ny Nx

// Error tolerence
#define ERROR_DECIMAL_DIG 1
#define ERROR 1e-1


// OMEGA is a parameter in the successive overrelaxation (SOR) technique, see eq. (17.31) in pp.383 of Landau.pdf
// Landau.pdf: A Survey of Computational Physics, Introductory Computational Science
#define OMEGA 1.0


// Boundary type
// 1: "linear", 2: "sinunoidal", any others: "constant"
#define BOUNDARY 2

#define DIRNAME "Datafile"
#define SUFFIX "_use_GS.csv"

// Following https://stackoverflow.com/a/5459929
#define STR_HELPER(INT) #INT
#define STR(INT) STR_HELPER(INT)

#define CONCATENATE(DIR, BD, SZ, DIG, SFX) DIRNAME "/" "boundary_" STR(BD) "_size_" STR(SZ) "_dig_" STR(DIG) SFX
#define FILENAME CONCATENATE(DIRNAME, BOUNDARY, Nx, ERROR_DECIMAL_DIG, SUFFIX)

// Thanks for Stefan Recksiegel to point it out that I can run tolerence_test every ten loops.
#define NLOOPS 1

#endif
