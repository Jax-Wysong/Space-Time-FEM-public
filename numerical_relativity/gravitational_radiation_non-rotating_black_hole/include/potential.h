#ifndef POTENTIAL_H
#define POTENTIAL_H

#include <petsc.h>
#include "appctx.h"

/* This computes the potential V_RW specifed in the paper*/
static inline PetscScalar V_RW(PetscScalar r, PetscInt ell, PetscScalar M)
{
    PetscScalar f   = 1.0 - 2.0*M / r;                 // Schwarzschild factor
    PetscScalar l2  = (PetscScalar)ell * (ell + 1);     // ell(ell+1)
    PetscScalar term= l2 / (r*r) - 6.0*M / (r*r*r);     // bracket
    return f * term;
}


#endif /* POTENTIAL_H */