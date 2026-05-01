#include "nonlin.h"

/* these compute the element-wise integrals of the nonlinear potential portion with gauss-quad */
void ComputeR_local_nonlinear(PetscScalar r_nonlin_u[4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, PetscReal x0, PetscReal t0, void *ctx)
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

	
    for (int i = 0; i < 4; ++i){
        r_nonlin_u[i] = 0.0;
	}

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            PetscReal xi  = gp[i];
            PetscReal tau = gp[j];
            PetscReal weight = gw[i] * gw[j];
            PetscReal x = 0.5 * (1.0 + xi) * hx;
            PetscReal t = 0.5 * (1.0 + tau) * ht;
            PetscReal detJ = 0.25 * hx * ht;
			

            PetscReal basis[4] = {
                (hx - x)*(ht - t)/(hx*ht),
                x*(ht - t)/(hx*ht),
                x*t/(hx*ht),
                (hx - x)*t/(hx*ht)
            };

			/* basis function derivatives wrt x */
			PetscReal dbasis_dx[4] = {
				-(ht - t)/(hx * ht),
				(ht - t)/(hx * ht),
				t/(hx * ht),
				-t/(hx * ht)
			};

			/* interpolate u */
			PetscScalar u_val = 0.0;
			for (int a = 0; a < 4; ++a){
				u_val += u_local[a] * basis[a];
			}

			/* compute du/dx */
			PetscScalar du_dx = 0.0;
			for (int a = 0; a < 4; ++a) {
				du_dx += u_local[a] * dbasis_dx[a];
			}



            PetscScalar u_nonlin = u_val * du_dx; /* for Burgers, nonlinear term is u * partial_x u */

            for (int a = 0; a < 4; ++a){
                r_nonlin_u[a] += basis[a] * u_nonlin * weight * detJ;
			}
		}
    }
}


void ComputeJ_local_nonlinear(PetscScalar J_local_u[4][4], const PetscScalar u_local[4], PetscReal hx, PetscReal ht, void *ctx)
{
		AppCtx *user = (AppCtx *)ctx;

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
				J_local_u[i][j] = 0.0;
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

				PetscReal dbasis_dx[4] = {
					-(ht - t)/(hx * ht),
					(ht - t)/(hx * ht),
					t/(hx * ht),
					-t/(hx * ht)
				};

				// Interpolate u and du/dx
				PetscScalar u_val = 0.0;
				PetscScalar du_dx = 0.0;
				for (int a = 0; a < 4; ++a) {
					u_val += u_local[a] * basis[a];
					du_dx += u_local[a] * dbasis_dx[a];
				}

				

						
				for (int a = 0; a < 4; ++a) {
					for (int b = 0; b < 4; ++b) {
						// product rule: d(u * u_x)/du_b = phi_b * u_x + u * dphi_b/dx
						J_local_u[a][b] += basis[a] * (basis[b] * du_dx + u_val * dbasis_dx[b]) * weight * detJ;
					}
				}
			}
		}
	
}