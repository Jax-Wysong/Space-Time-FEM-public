#include "source.h"
#include "appctx.h"
#include "compute_rstar.h"

/* 
    this is where we compute the RHS integral of the source term.
    Since we already analytically integrated in in space with the help
    of the delta function, we only need to integrate wrt time.
*/
void ComputeR_local_pp(PetscScalar r_pp[4], PetscReal hx, PetscReal ht, PetscReal x0, PetscReal t0, void *ctx)
{
    AppCtx *user = (AppCtx*)ctx;

    // zero out
    for (int a=0; a<4; ++a) r_pp[a] = 0.0;

    // 1D Gauss-Legendre in time 
	const PetscReal gp[4] = {
	  -0.8611363116, -0.3399810436,
	   0.3399810436,  0.8611363116
	};
	
	const PetscReal gw[4] = {
	   0.3478548451,  0.6521451549,
	   0.6521451549,  0.3478548451
	};

    // map tau in [-1,1] -> t in [t0,t1]
    PetscReal t1 = t0 + ht;
    PetscReal dt = 0.5 * (t1 - t0);

    for (int q=0; q<4; ++q) {
        PetscReal tau = gp[q];
        PetscReal w   = gw[q];

        PetscReal t   = 0.5 * ( (1.0 - tau)*t0 + (1.0 + tau)*t1 ); // equivalently t0 + (tau+1)/2*ht

        // particle position at this time
        PetscReal x_p = worldline_xp(t, user);

        // does the particle lie in [x0,x0+hx] at this time?
        if (x_p < x0 || x_p > x0 + hx) continue;

        // local x coordinate within element
        PetscReal x_loc = x_p - x0;   // in [0,hx]

        // local t coordinate within element
        PetscReal t_loc = t - t0;     // in [0,ht]

        // Q1 basis and ∂/∂x at (x_loc, t_loc)
        PetscReal N[4], dNdx[4];

        N[0] = (hx - x_loc)*(ht - t_loc)/(hx*ht);
        N[1] =  x_loc      *(ht - t_loc)/(hx*ht);
        N[2] =  x_loc      * t_loc      /(hx*ht);
        N[3] = (hx - x_loc)* t_loc      /(hx*ht);

        dNdx[0] = -(ht - t_loc)/(hx*ht);
        dNdx[1] =  (ht - t_loc)/(hx*ht);
        dNdx[2] =   t_loc      /(hx*ht);
        dNdx[3] =  -t_loc      /(hx*ht);

        // coefficients tildeF(t), tildeG(t)
        PetscScalar Gtilde = TildeG(t, user);  // from Sopuerta formulas
        PetscScalar Ftilde = TildeF(t, user);  // from Sopuerta formulas

        PetscReal weight = w * dt;  // 1D time integration

        for (int a=0; a<4; ++a) {
            r_pp[a] += -1.0*( Gtilde * N[a] - Ftilde * dNdx[a] ) * weight;
        }
    }
}


void ComputeR_local_source(PetscScalar r_source_u[4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, PetscReal x0, PetscReal t0, void *ctx)
{
	PetscReal xR, xL;  
	AppCtx *user = (AppCtx *)ctx;
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

    PetscReal sigma = 3.0 * user->hx;  // width of fake particle
    PetscReal x_p   = 10.0;            // static fake particle
    PetscReal q     = 1.0;

    for (int i = 0; i < 4; ++i){
        r_source_u[i] = 0.0;
	}

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            PetscReal xi  = gp[i];
            PetscReal tau = gp[j];
            PetscReal weight = gw[i] * gw[j];
            PetscReal x = 0.5 * (1.0 + xi) * hx;
            PetscReal t = 0.5 * (1.0 + tau) * ht;
            PetscReal detJ = 0.25 * hx * ht;
			
			// global coords for source function
			PetscReal x_phys = x0 + x;
			PetscReal t_phys = t0 + t;
			

            PetscReal basis[4] = {
                (hx - x)*(ht - t)/(hx*ht),
                x*(ht - t)/(hx*ht),
                x*t/(hx*ht),
                (hx - x)*t/(hx*ht)
            };
			

            // --- fake particle source F(t,x) = q * Gaussian(x - x_p)
            PetscReal dx   = x_phys - x_p;       // if you want x_p(t), use x_p(t_phys)
            PetscReal norm = 1.0 / (PetscSqrtReal(2.0*PETSC_PI) * sigma);
            PetscReal G    = norm * PetscExpReal( - (dx*dx) / (2.0*sigma*sigma) );
            PetscScalar source = q * G;

            for (int a = 0; a < 4; ++a){
                r_source_u[a] += basis[a] * (-1.0*source) * weight * detJ;
			}
		}
    }
}


void ComputeJ_local_source(PetscScalar J_local_u[4][4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, void *ctx)
{
/*	
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
*/
}