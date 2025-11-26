#include "nonlin.h"
#include "potential.h"

void ComputeR_local_nonlinear(PetscScalar r_nonlin_phi[8], PetscScalar r_nonlin_chi[8],
                              const PetscScalar phi_local[8], const PetscScalar chi_local[8],
                              PetscReal hx, PetscReal hy, PetscReal ht, void *ctx)
{
    AppCtx *user = (AppCtx *)ctx;

	const PetscReal gp[4] = {
	  -0.8611363116, -0.3399810436,
	   0.3399810436,  0.8611363116
	};
	
	const PetscReal gw[4] = {
	   0.3478548451,  0.6521451549,
	   0.6521451549,  0.3478548451
	};

    for (int i = 0; i < 8; ++i) {
        r_nonlin_phi[i] = 0.0;
        r_nonlin_chi[i] = 0.0;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                PetscReal xi  = gp[i];
                PetscReal eta = gp[j];
                PetscReal tau = gp[k];
                PetscReal weight = gw[i] * gw[j] * gw[k];
                PetscReal x = 0.5 * (1.0 + xi) * hx;
                PetscReal y = 0.5 * (1.0 + eta) * hy;
                PetscReal t = 0.5 * (1.0 + tau) * ht;
                PetscReal detJ = (1.0/8.0) * hx * hy * ht;

                PetscReal basis[8] = {
                    (hx - x)*(hy - y)*(ht - t)/(hx*hy*ht),
                    x*(hy - y)*(ht - t)/(hx*hy*ht),
                    x*y*(ht - t)/(hx*hy*ht),
                    (hx - x)*y*(ht - t)/(hx*hy*ht),
                    (hx - x)*(hy - y)*t/(hx*hy*ht),
                    x*(hy - y)*t/(hx*hy*ht),
                    x*y*t/(hx*hy*ht),
                    (hx - x)*y*t/(hx*hy*ht)
                };

                PetscScalar phi_val = 0.0;
                PetscScalar chi_val = 0.0;
                for (int a = 0; a < 8; ++a){
                    phi_val += phi_local[a] * basis[a];
                    chi_val += chi_local[a] * basis[a];
                }

                PetscScalar V, V_phi, V_chi;
                PetscScalar V_phiphi, V_phichi, V_chiphi, V_chichi;

                compute_potential(phi_val, chi_val, user,
                                  &V,
                                  &V_phi, &V_chi,
                                  &V_phiphi, &V_phichi,
                                  &V_chiphi, &V_chichi);

                for (int a = 0; a < 8; ++a){
                    r_nonlin_phi[a] += basis[a] * V_phi * weight * detJ;
                    r_nonlin_chi[a] += basis[a] * (user->ghost) * V_chi * weight * detJ;
                }
            }
        }
    }
}

void ComputeJ_local_nonlinear(PetscScalar J_local_pp[8][8], PetscScalar J_local_pc[8][8],
                              PetscScalar J_local_cc[8][8], PetscScalar J_local_cp[8][8],
                              const PetscScalar phi_local[8], const PetscScalar chi_local[8],
                              PetscReal hx, PetscReal hy, PetscReal ht, void *ctx)
{
    AppCtx *user = (AppCtx *)ctx;

	const PetscReal gp[4] = {
	  -0.8611363116, -0.3399810436,
	   0.3399810436,  0.8611363116
	};
	
	const PetscReal gw[4] = {
	   0.3478548451,  0.6521451549,
	   0.6521451549,  0.3478548451
	};

    for (int i = 0; i < 8; ++i){
        for (int j = 0; j < 8; ++j){
            J_local_pp[i][j] = 0.0;
            J_local_pc[i][j] = 0.0;
            J_local_cc[i][j] = 0.0;
            J_local_cp[i][j] = 0.0;
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                PetscReal xi  = gp[i];
                PetscReal eta = gp[j];
                PetscReal tau = gp[k];
                PetscReal weight = gw[i] * gw[j] * gw[k];
                PetscReal x = 0.5 * (1.0 + xi) * hx;
                PetscReal y = 0.5 * (1.0 + eta)* hy;
                PetscReal t = 0.5 * (1.0 + tau) * ht;
                PetscReal detJ = (1.0/8.0) * hx * hy * ht;

                PetscReal basis[8] = {
                    (hx - x)*(hy - y)*(ht - t)/(hx*hy*ht),
                    x*(hy - y)*(ht - t)/(hx*hy*ht),
                    x*y*(ht - t)/(hx*hy*ht),
                    (hx - x)*y*(ht - t)/(hx*hy*ht),
                    (hx - x)*(hy - y)*t/(hx*hy*ht),
                    x*(hy - y)*t/(hx*hy*ht),
                    x*y*t/(hx*hy*ht),
                    (hx - x)*y*t/(hx*hy*ht)
                };

                PetscScalar phi_val = 0.0;
                PetscScalar chi_val = 0.0;
                for (int a = 0; a < 8; ++a){
                    phi_val += phi_local[a] * basis[a];
                    chi_val += chi_local[a] * basis[a];
                }

                PetscScalar V, V_phi, V_chi;
                PetscScalar V_phiphi, V_phichi, V_chiphi, V_chichi;

                compute_potential(phi_val, chi_val, user,
                                  &V,
                                  &V_phi, &V_chi,
                                  &V_phiphi, &V_phichi,
                                  &V_chiphi, &V_chichi);

                for (int a = 0; a < 8; ++a) {
                    for (int b = 0; b < 8; ++b) {
                        J_local_pp[a][b] += basis[a] * basis[b] * V_phiphi * weight * detJ;
                        J_local_pc[a][b] += basis[a] * basis[b] * V_phichi * weight * detJ;
                        J_local_cc[a][b] += basis[a] * basis[b] * (user->ghost)*V_chichi * weight * detJ;
                        J_local_cp[a][b] += basis[a] * basis[b] * (user->ghost)*V_chiphi * weight * detJ;
                    }
                }
            }
        }
    }
}