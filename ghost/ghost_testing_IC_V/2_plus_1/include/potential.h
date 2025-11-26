#ifndef POTENTIAL_H
#define POTENTIAL_H

#include "nonlin.h"

/* 
 * These functions compute:
 *  V
 *  V_phi   
 *  V_chi   
 *  V_phiphi
 *  V_phichi
 *  V_chiphi
 *  V_chichi
 */

/* ----------------- lam22 potential: V = phi^2 chi^2 ----------------- */
static inline void
lam22_potential(PetscScalar phi, PetscScalar chi, const AppCtx *user,
                PetscScalar *V,
                PetscScalar *V_phi,    PetscScalar *V_chi,
                PetscScalar *V_phiphi, PetscScalar *V_phichi,
                PetscScalar *V_chiphi, PetscScalar *V_chichi)
{
    const PetscReal lam = user->lam22;

    *V = lam*phi*phi*chi*chi;

    /* First derivatives: */
    *V_phi = lam * 2.0 * phi * (chi * chi);     
    *V_chi = lam * 2.0 * chi * (phi * phi);     

    /* Second derivatives: */
    *V_phiphi = lam * 2.0 * (chi * chi);        
    *V_phichi = lam * 4.0 * phi * chi;          
    *V_chiphi = lam * 4.0 * phi * chi;          
    *V_chichi = lam * 2.0 * (phi * phi);        
}

/* ----------------- phi^6 potential: e.g. V = 1/2 m^2 W - lambda/4 W^2 + g/6 W^3 (W = phi^2 + chi^2)  ----------- */
static inline void
lam_phi6_potential(PetscScalar phi, PetscScalar chi, const AppCtx *user,
                   PetscScalar *V,
                   PetscScalar *V_phi,    PetscScalar *V_chi,
                   PetscScalar *V_phiphi, PetscScalar *V_phichi,
                   PetscScalar *V_chiphi, PetscScalar *V_chichi)
{
    const PetscReal lambda = user->lambda_66;
    const PetscReal g66    = user->g66;
    const PetscReal m2     = user->mphi2;

    PetscScalar W = phi*phi + chi*chi;
    *V = 0.5 * m2 * W - lambda/4 * W*W + g66/6 * W*W*W;

	*V_phi = phi*(g66*PetscPowReal(PetscPowReal(chi, 2) + PetscPowReal(phi, 2), 2) - lambda*(PetscPowReal(chi, 2) + PetscPowReal(phi, 2)) + m2);
	*V_chi = chi*(g66*PetscPowReal(PetscPowReal(chi, 2) + PetscPowReal(phi, 2), 2) - lambda*(PetscPowReal(chi, 2) + PetscPowReal(phi, 2)) + m2);

    *V_phiphi = PetscPowReal(chi, 4)*g66 + 6*PetscPowReal(chi, 2)*g66*PetscPowReal(phi, 2) - PetscPowReal(chi, 2)*lambda + 5*g66*PetscPowReal(phi, 4) - 3*lambda*PetscPowReal(phi, 2) + m2;
	*V_phichi = 2*chi*phi*(2*g66*(PetscPowReal(chi, 2) + PetscPowReal(phi, 2)) - lambda);
	*V_chichi = 5*PetscPowReal(chi, 4)*g66 + 6*PetscPowReal(chi, 2)*g66*PetscPowReal(phi, 2) - 3*PetscPowReal(chi, 2)*lambda + g66*PetscPowReal(phi, 4) - lambda*PetscPowReal(phi, 2) + m2;
    *V_chiphi = 2*chi*phi*(2*g66*(PetscPowReal(chi, 2) + PetscPowReal(phi, 2)) - lambda);

}

/* ------------- choose which potential to use -------- */

static inline void
compute_potential(PetscScalar phi, PetscScalar chi, const AppCtx *user,
                  PetscScalar *V,
                  PetscScalar *V_phi,    PetscScalar *V_chi,
                  PetscScalar *V_phiphi, PetscScalar *V_phichi,
                  PetscScalar *V_chiphi, PetscScalar *V_chichi)
{
    if (user->lam22 == 1.0 && user->lam_phi6 == 0.0) {
        lam22_potential(phi, chi, user,
                        V,
                        V_phi, V_chi,
                        V_phiphi, V_phichi,
                        V_chiphi, V_chichi);
    } else if (user->lam22 == 0.0 && user->lam_phi6 == 1.0) {
        lam_phi6_potential(phi, chi, user,
                           V,
                           V_phi, V_chi,
                           V_phiphi, V_phichi,
                           V_chiphi, V_chichi);
    } else {
        /* Fallback: no potential or misconfigured flags */
        *V = 0.0;
        *V_phi = *V_chi = 0.0;
        *V_phiphi = *V_phichi = *V_chiphi = *V_chichi = 0.0;
    }
}

#endif /* POTENTIAL_H */
