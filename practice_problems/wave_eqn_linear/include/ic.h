#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>

/* ======================= Wave -IC 1 ======================= */
static inline PetscScalar wave_IC_u(PetscReal x)
{	
  return PetscExpReal(-(x - 1.0) * (x - 1.0));
} 

static inline PetscScalar wave_IC_v(PetscReal x)
{	
  (void)x;
  return 0.0;
} 

#endif /* IC_H */