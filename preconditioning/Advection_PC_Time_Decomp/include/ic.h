#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>

/* ======================= Gaussian Bump -IC 1 ======================= */
static inline PetscScalar Gaussian_IC_u(PetscReal x)
{	
  PetscReal x0 = 0.5; /* center of Gaussian */
  PetscReal sigma = 0.1; /* width of Gaussian */
  PetscReal A = 1.0; /* amplitude of Gaussian */
  PetscScalar u_0 = A*PetscExpScalar(-PetscPowScalar(x-x0,2)/(2*PetscPowScalar(sigma,2)));
  return u_0;
} 



#endif /* IC_H */