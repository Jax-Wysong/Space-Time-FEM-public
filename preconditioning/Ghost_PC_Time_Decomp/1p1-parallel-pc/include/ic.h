#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>

/* ======================= MMS -IC 1 ======================= */
static inline PetscScalar phi_MMS_IC(PetscReal x)
{	
  return 1.0 + PetscCosReal(PETSC_PI*x);
} 

static inline PetscScalar u_MMS_IC(PetscReal x)
{
	return 0.0;
}

static inline PetscScalar chi_MMS_IC(PetscReal x)
{
	return 1.0;
}

static inline PetscScalar v_MMS_IC(PetscReal x)
{
	return PETSC_PI * PetscCosReal(PETSC_PI*x);
}


#endif /* IC_H */