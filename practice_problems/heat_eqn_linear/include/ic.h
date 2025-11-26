#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>

/* ======================= Heat Linear -IC 1 ======================= */
static inline PetscScalar heat_linear_IC(PetscReal x)
{	
  return PetscSinReal(PETSC_PI*x);
} 

#endif /* IC_H */