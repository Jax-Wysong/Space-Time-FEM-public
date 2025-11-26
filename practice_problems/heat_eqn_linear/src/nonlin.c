#include "nonlin.h"

/* nothing is used here for the homogenous heat_linear problem */
void ComputeR_local_nonlinear(PetscScalar r_nonlin_phi[4], PetscScalar r_nonlin_chi[4], const PetscScalar phi_local[4], const PetscScalar chi_local[4], PetscReal hx, PetscReal ht, PetscReal x0, PetscReal t0, void *ctx)
{
	PetscReal lam22, ghost, xR, xL;  
	AppCtx *user = (AppCtx *)ctx;
	lam22 = user->lam22;
	ghost = user->ghost;
	xR = user->xR;
	xL = user->xL;
  /*	
    const PetscReal gp[2] = { -1.0 / PetscSqrtReal(3.0), 1.0 / PetscSqrtReal(3.0) };
      const PetscReal gw[2] = { 1.0, 1.0 };	
  */	
	const PetscReal gp[4] = {
	  -0.8611363116, -0.3399810436,
	   0.3399810436,  0.8611363116
	};
	
	const PetscReal gw[4] = {
	   0.3478548451,  0.6521451549,
	   0.6521451549,  0.3478548451
	};

	
    for (int i = 0; i < 4; ++i){
        r_nonlin_phi[i] = 0.0;
		r_nonlin_chi[i] = 0.0;
	}

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            PetscReal xi  = gp[i];
            PetscReal tau = gp[j];
            PetscReal weight = gw[i] * gw[j];
            PetscReal x = 0.5 * (1.0 + xi) * hx;
            PetscReal t = 0.5 * (1.0 + tau) * ht;
            PetscReal detJ = 0.25 * hx * ht;
			
			// global coords for forcing function
			PetscReal L = xR - xL;
			PetscReal raw = (x0 + x - xL);
			PetscReal x_phys = xL + PetscFmodReal(raw + L, L);
			PetscReal t_phys = t0 + t;
			

            PetscReal basis[4] = {
                (hx - x)*(ht - t)/(hx*ht),
                x*(ht - t)/(hx*ht),
                x*t/(hx*ht),
                (hx - x)*t/(hx*ht)
            };
			

            PetscScalar phi_val = 0.0;
			PetscScalar chi_val = 0.0;
            for (int a = 0; a < 4; ++a){
                phi_val += phi_local[a] * basis[a];
				chi_val += chi_local[a] * basis[a];				
			}
			
			 PetscScalar A = (phi_val*phi_val) - (chi_val*chi_val) + 1.0;
			 PetscScalar C = (phi_val*phi_val) - (chi_val*chi_val) - 1.0;
			 PetscScalar B = C*C + 4.0*phi_val*phi_val;
            PetscScalar V_phi = -2.0*lam22*phi_val*(A)*PetscPowScalar(B, -1.5);
			PetscScalar V_chi = 2.0*lam22*chi_val*(C)*PetscPowScalar(B, -1.5);
			
			/////// Set up Forcing term here F_phi(x,t) = phi(x,t) + V_phi(phi(x,t), chi(x,t) //////
			
			////// phi-xt = 1 + PetscSinReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * t_phys) /////
			////// chi-xt = 1 + PetscCosReal(PETSC_PI * x_phys) * PetscSinReal(PETSC_PI * t_phys) /////
			PetscScalar phixt = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * t_phys);
			PetscScalar chixt = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscSinReal(PETSC_PI * t_phys);
			PetscScalar A_f = (phixt*phixt) - (chixt*chixt) + 1.0;
			PetscScalar C_f = (phixt*phixt) - (chixt*chixt) - 1.0;
			PetscScalar B_f = C_f*C_f + 4.0*phixt*phixt;
			
			PetscScalar forcing_phi = phixt + (-2.0) * lam22 * phixt * A_f * PetscPowScalar(B_f, -1.5);
			PetscScalar forcing_chi = chixt - 2.0 * lam22 * chixt * C_f * PetscPowScalar(B_f, -1.5);
			  			
            for (int a = 0; a < 4; ++a){
                r_nonlin_phi[a] += basis[a] * V_phi * weight * detJ - (basis[a] * forcing_phi * weight * detJ);
				r_nonlin_chi[a] += basis[a] * ghost*V_chi * weight * detJ - (basis[a] * forcing_chi * weight * detJ);
			}
		}
    }
}


void ComputeJ_local_nonlinear(PetscScalar J_local_pp[4][4], PetscScalar J_local_pc[4][4], PetscScalar J_local_cc[4][4], PetscScalar J_local_cp[4][4], const PetscScalar phi_local[4], const PetscScalar chi_local[4], PetscReal hx, PetscReal ht, void *ctx)
{
	
	AppCtx *user = (AppCtx *)ctx;
	PetscReal lam22 = user->lam22;
	PetscReal ghost = user->ghost;

  //	const PetscReal gp[2] = { -1.0 / PetscSqrtReal(3.0), 1.0 / PetscSqrtReal(3.0) };
  //   const PetscReal gw[2] = { 1.0, 1.0 };	
    
	const PetscReal gp[4] = {
	  -0.8611363116, -0.3399810436,
	   0.3399810436,  0.8611363116
	};
	
	const PetscReal gw[4] = {
	   0.3478548451,  0.6521451549,
	   0.6521451549,  0.3478548451
	};


   for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            J_local_pp[i][j] = 0.0;
			J_local_pc[i][j] = 0.0;
			J_local_cc[i][j] = 0.0;
			J_local_cp[i][j] = 0.0;
		}
	}

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            PetscReal xi  = gp[i];
            PetscReal tau = gp[j];
            PetscReal weight = gw[i] * gw[j];
            PetscReal x = 0.5 * (1.0 + xi) * hx;
            PetscReal t = 0.5 * (1.0 + tau) * ht;
            PetscReal detJ = 0.25 * hx * ht;

            // Evaluate basis functions
            PetscReal basis[4] = {
                (hx - x)*(ht - t)/(hx*ht),
                x*(ht - t)/(hx*ht),
                x*t/(hx*ht),
                (hx - x)*t/(hx*ht)
            };

            // Interpolate u
            PetscScalar phi_val = 0.0;
			PetscScalar chi_val = 0.0;
            for (int a = 0; a < 4; ++a){
                phi_val += phi_local[a] * basis[a];
				chi_val += chi_local[a] * basis[a];
			}
			
			//// Derivative of source term ////
			 PetscScalar A = (phi_val*phi_val) - (chi_val*chi_val) + 1.0;
			 PetscScalar C = (phi_val*phi_val) - (chi_val*chi_val) - 1.0;
			 PetscScalar B = C*C + 4.0*phi_val*phi_val;
			 
			PetscScalar V_phiphi = (-2.0*lam22*PetscPowScalar(B, -2.5)) * ((3.0*phi_val*phi_val - chi_val*chi_val + 1.0)*(B) - 6.0*phi_val*phi_val*(A*A));
			PetscScalar V_phichi = 4.0*lam22*phi_val*chi_val*(PetscPowScalar(B, -2.5))*(B - 3.0*A*C);
			PetscScalar V_chichi = 2.0*lam22*((phi_val*phi_val - 3.0*chi_val*chi_val - 1) * PetscPowScalar(B, -1.5) + (6.0*chi_val*chi_val*(C*C))*PetscPowScalar(B, -2.5));
            PetscScalar V_chiphi = 4.0*lam22*phi_val*chi_val*(PetscPowScalar(B, -2.5))*(B - 3.0*C*A);

					
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    J_local_pp[a][b] += basis[a] * basis[b] * V_phiphi * weight * detJ;
					J_local_pc[a][b] += basis[a] * basis[b] * V_phichi * weight * detJ;
					J_local_cc[a][b] += basis[a] * basis[b] * ghost*V_chichi * weight * detJ;
					J_local_cp[a][b] += basis[a] * basis[b] * ghost*V_chiphi * weight * detJ;
					
                }
            }
        }
    }
}