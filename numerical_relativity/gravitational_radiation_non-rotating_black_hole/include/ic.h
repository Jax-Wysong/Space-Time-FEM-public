#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>
#include "appctx.h"


/* ======================= -IC 1 ======================= */
static inline PetscScalar BH_IC_u(PetscReal x)
{	
  (void)x;
  return 0.0;
} 

static inline PetscScalar BH_IC_v(PetscReal x)
{	
  (void)x;
  return 0.0;
} 

/* ======================== F and G tilde ================ */
static inline PetscScalar TildeG(PetscReal t, const AppCtx *user)
{
  PetscReal q = 1.0;
    return q;   // constant charge
}

static inline PetscScalar TildeF(PetscReal t, const AppCtx *user)
{
    PetscReal qF = 0.0;
    return qF;       // ignore δ' for first test
}


/* ======================================
   Gaussian initial data for RW test
   ====================================== */

static inline PetscScalar Gauss_IC_u(PetscReal x)
{
    PetscReal A     = 1.0;
    PetscReal x0    = 50.0;
    PetscReal sigma = 3.0;

    PetscReal dx = x - x0;
    return A * PetscExpReal(-(dx*dx)/(2.0*sigma*sigma));
}

static inline PetscScalar Gauss_IC_v(PetscReal x)
{
    PetscReal A     = 1.0;
    PetscReal x0    = 50.0;
    PetscReal sigma = 3.0;

    PetscReal dx = x - x0;
    PetscScalar u0 = A * PetscExpReal(-(dx*dx)/(2.0*sigma*sigma));

    // Speed = +1 or -1 depending on direction
    PetscReal sign = -1.0;   // negative: moves toward smaller x (BH direction)

    return sign * (dx/(sigma*sigma)) * u0;
}



#endif /* IC_H */