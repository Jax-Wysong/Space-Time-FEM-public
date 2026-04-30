#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>

/* ======================= Burgers -IC 1 ======================= */
static inline PetscScalar Burgers_IC_u(PetscReal x, PetscReal viscosity)
{	
  PetscReal psi_0 = PetscExpReal(-(x)*(x)/(4*viscosity)) + PetscExpReal(-(x-2*PETSC_PI)*(x-2*PETSC_PI)/(4*viscosity));
  PetscReal psi_x_0 = -(2*x)*PetscExpReal(-(x)*(x)/(4*viscosity))/(4*viscosity) - (2*x - 4*PETSC_PI)*PetscExpReal(-(x-2*PETSC_PI)*(x-2*PETSC_PI)/(4*viscosity))/(4*viscosity);
  PetscReal u_0 = -2.0*viscosity*psi_x_0/psi_0 + 4.0;
  return u_0;
} 



#endif /* IC_H */