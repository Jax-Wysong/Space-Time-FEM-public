#include "nonlin.h"

static void evaluate_basis(PetscReal hx, PetscReal hy, PetscReal ht, PetscReal x, PetscReal y, PetscReal t, PetscScalar basis[8])
{
    basis[0] = (hx - x)*(hy - y)*(ht - t)/(hx*hy*ht);
    basis[1] = x*(hy - y)*(ht - t)/(hx*hy*ht);
    basis[2] = x*y*(ht - t)/(hx*hy*ht); 
    basis[3] = (hx - x)*y*(ht - t)/(hx*hy*ht);
    basis[4] = (hx - x)*(hy - y)*t/(hx*hy*ht);
    basis[5] = x*(hy - y)*t/(hx*hy*ht);
    basis[6] = x*y*t/(hx*hy*ht); 
    basis[7] = (hx - x)*y*t/(hx*hy*ht); 
}

void ComputeR_local_nonlinear_phi(PetscScalar r_nonlin[8], const PetscScalar phi_local[8], const PetscScalar chi_local[8], PetscReal hx, PetscReal hy, PetscReal ht, PetscReal x0, PetscReal y0, PetscReal t0, void *ctx)
{
        PetscReal lam, xL, xR, yL, yR; 
        AppCtx *user = (AppCtx *)ctx;
        lam = user->lam;
        xL = user->xL;
        xR = user->xR;
        yL = user->yL;
        yR = user->yR;

        const PetscReal gp[2] = { -1.0 / PetscSqrtReal(3.0), 1.0 / PetscSqrtReal(3.0) };
        const PetscReal gw[2] = { 1.0, 1.0 };

	// const PetscReal gp[4] = {
	//   -0.8611363116, -0.3399810436,
	//    0.3399810436,  0.8611363116
	// };
	
	// const PetscReal gw[4] = {
	//    0.3478548451,  0.6521451549,
	//    0.6521451549,  0.3478548451
	// };

        for (int i = 0; i < 8; ++i)
                r_nonlin[i] = 0.0;

        for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                        for (int k = 0; k < 2; ++k) {
                                PetscReal xi  = gp[i];
                                PetscReal eta = gp[j];
                                PetscReal tau = gp[k];
                                PetscReal weight = gw[i] * gw[j] * gw[k];
                                PetscReal x = 0.5 * (1.0 + xi) * hx;
                                PetscReal y = 0.5 * (1.0 + eta) * hy;
                                PetscReal t = 0.5 * (1.0 + tau) * ht;
                                PetscReal detJ = (1.0/8.0) * hx * hy * ht;

                                // global coords for forcing function
                                PetscReal Lx = xR - xL;
                                PetscReal rawx = (x0 + x - xL);
                                PetscReal x_phys = xL + PetscFmodReal(rawx + Lx, Lx);
                                PetscReal Ly = yR - yL;
                                PetscReal rawy = (y0 + y - yL);
                                PetscReal y_phys = yL + PetscFmodReal(rawy + Ly, Ly);
                                PetscReal t_phys = t0 + t;

                                PetscScalar basis[8];
                                evaluate_basis(hx, hy, ht, x, y, t, basis);
                                PetscScalar phi_val = 0.0;
                                PetscScalar chi_val = 0.0;
                                for (int a = 0; a < 8; ++a){
                                        phi_val += phi_local[a] * basis[a];
                                        chi_val += chi_local[a] * basis[a];
                                }


                                 PetscScalar A = (phi_val*phi_val) - (chi_val*chi_val) + 1.0;
                                 PetscScalar C = (phi_val*phi_val) - (chi_val*chi_val) - 1.0;
                                 PetscScalar B = C*C + 4.0*phi_val*phi_val;
                                PetscScalar V_phi = -2.0*lam*phi_val*(A)*PetscPowScalar(B, -1.5);

                                /////// Set up Forcing term here F_phi(x,t) = phi(x,t) + V_phi(phi(x,t), chi(x,t) //////

                                ////// phi-xt = 1 + PetscSinReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * t_phys) /////
                                ////// chi-xt = 1 + PetscCosReal(PETSC_PI * x_phys) * PetscSinReal(PETSC_PI * t_phys) /////
                                PetscScalar phixt = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys) * PetscCosReal(PETSC_PI * t_phys);
                                PetscScalar chixt = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys) * PetscSinReal(PETSC_PI * t_phys);
                                PetscScalar A_f = (phixt*phixt) - (chixt*chixt) + 1.0;
                                PetscScalar C_f = (phixt*phixt) - (chixt*chixt) - 1.0;
                                PetscScalar B_f = C_f*C_f + 4.0*phixt*phixt;
                                PetscScalar forcing_phi = (PETSC_PI*PETSC_PI * PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys) * PetscCosReal(PETSC_PI * t_phys)) + phixt + (-2.0) * lam * phixt * A_f * PetscPowScalar(B_f, -1.5);

                                for (int a = 0; a < 8; ++a)
                                        r_nonlin[a] += basis[a] * V_phi * weight * detJ - (basis[a] * forcing_phi * weight * detJ);

                        }
                }
        }
}


void ComputeR_local_nonlinear_chi(PetscScalar r_nonlin[8], const PetscScalar chi_local[8], const PetscScalar phi_local[8], PetscReal hx, PetscReal hy, PetscReal ht, PetscReal x0, PetscReal y0, PetscReal t0, void *ctx)
{
        PetscReal lam, xL, xR, yL, yR;  //lambda damping constant
        AppCtx *user = (AppCtx *)ctx;
        lam = user->lam;
        xL = user->xL;
        xR = user->xR;
        yL = user->yL;
        yR = user->yR;

        const PetscReal gp[2] = { -1.0 / PetscSqrtReal(3.0), 1.0 / PetscSqrtReal(3.0) };
        const PetscReal gw[2] = { 1.0, 1.0 };

	// const PetscReal gp[4] = {
	//   -0.8611363116, -0.3399810436,
	//    0.3399810436,  0.8611363116
	// };
	
	// const PetscReal gw[4] = {
	//    0.3478548451,  0.6521451549,
	//    0.6521451549,  0.3478548451
	// };

         for (int i = 0; i < 8; ++i)
                r_nonlin[i] = 0.0;

        for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                        for (int k = 0; k < 2; ++k) {
                                PetscReal xi  = gp[i];
                                PetscReal eta  = gp[j];
                                PetscReal tau = gp[k];
                                PetscReal weight = gw[i] * gw[j] * gw[k];
                                PetscReal x = 0.5 * (1.0 + xi) * hx;
                                PetscReal y = 0.5 * (1.0 + eta) * hy;
                                PetscReal t = 0.5 * (1.0 + tau) * ht;
                                PetscReal detJ = (1.0/8.0) * hx * hy* ht;

                                // global coords for forcing function
                                PetscReal L = xR - xL;
                                PetscReal raw = (x0 + x - xL);
                                PetscReal x_phys = xL + PetscFmodReal(raw + L, L);
                                PetscReal Ly = yR - yL;
                                PetscReal rawy = (y0 + y - yL);
                                PetscReal y_phys = yL + PetscFmodReal(rawy + Ly, Ly);
                                PetscReal t_phys = t0 + t;

                                PetscScalar basis[8];
                                evaluate_basis(hx, hy, ht, x, y, t, basis);


                                PetscScalar chi_val = 0.0;
                                PetscScalar phi_val = 0.0;
                                for (int a = 0; a < 8; ++a){
                                        chi_val += chi_local[a] * basis[a];
                                        phi_val += phi_local[a] * basis[a];
                                }


                                 PetscScalar C = (phi_val*phi_val) - (chi_val*chi_val) - 1.0;
                                 PetscScalar B = C*C + 4.0*phi_val*phi_val;
                                PetscScalar V_chi = 2.0*lam*chi_val*(C)*PetscPowScalar(B, -1.5);

                                /////// Set up Forcing term here F_chi(x,t) = chi(x,t) - V_chi(phi(x,t), chi(x,t) //////

                                ////// phi-xt = 1 + PetscSinReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * t_phys) /////
                                ////// chi-xt = 1 + PetscCosReal(PETSC_PI * x_phys) * PetscSinReal(PETSC_PI * t_phys) /////
                                PetscScalar phixt = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys) * PetscCosReal(PETSC_PI * t_phys);
                                PetscScalar chixt = 1.0 + PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys) * PetscSinReal(PETSC_PI * t_phys);
                                PetscScalar C_f = (phixt*phixt) - (chixt*chixt) - 1.0;
                                PetscScalar B_f = C_f*C_f + 4.0*phixt*phixt;
                                PetscScalar forcing_chi = (PETSC_PI*PETSC_PI*PetscCosReal(PETSC_PI * x_phys) * PetscCosReal(PETSC_PI * y_phys) * PetscSinReal(PETSC_PI * t_phys)) + chixt - 2.0 * lam * chixt * C_f * PetscPowScalar(B_f, -1.5);

                                for (int a = 0; a < 8; ++a)
                                        r_nonlin[a] += basis[a] * -1.0*V_chi * weight * detJ - (basis[a] * forcing_chi * weight * detJ);
                        }
                }
        }
}


void ComputeJ_local_nonlinear(PetscScalar J_local_pp[8][8], PetscScalar J_local_pc[8][8], PetscScalar J_local_cc[8][8], PetscScalar J_local_cp[8][8], const PetscScalar phi_local[8], const PetscScalar chi_local[8], PetscReal hx, PetscReal hy, PetscReal ht, void *ctx)
{
    const PetscReal gp[2] = { -1.0 / PetscSqrtReal(3.0), 1.0 / PetscSqrtReal(3.0) };
    const PetscReal gw[2] = { 1.0, 1.0 };

	// const PetscReal gp[4] = {
	//   -0.8611363116, -0.3399810436,
	//    0.3399810436,  0.8611363116
	// };
	
	// const PetscReal gw[4] = {
	//    0.3478548451,  0.6521451549,
	//    0.6521451549,  0.3478548451
	// };
    
    for (int i = 0; i < 8; ++i){
        for (int j = 0; j < 8; ++j){
                J_local_pp[i][j] = 0.0;
                J_local_pc[i][j] = 0.0;
                J_local_cc[i][j] = 0.0;
                J_local_cp[i][j] = 0.0;
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                        PetscReal xi  = gp[i];
                        PetscReal eta = gp[j];
                        PetscReal tau = gp[k];
                        PetscReal weight = gw[i] * gw[j] * gw[k];
                        PetscReal x = 0.5 * (1.0 + xi) * hx;
                        PetscReal y = 0.5 * (1.0 + eta)* hy;
                        PetscReal t = 0.5 * (1.0 + tau) * ht;
                        PetscReal detJ = (1.0/8.0) * hx * hy * ht;

                        // Evaluate basis functions

                        PetscScalar basis[8];
                        evaluate_basis(hx, hy, ht, x, y, t, basis);


                        // Interpolate u
                        PetscScalar phi_val = 0.0;
                        PetscScalar chi_val = 0.0;
                        for (int a = 0; a < 8; ++a){
                                phi_val += phi_local[a] * basis[a];
                                chi_val += chi_local[a] * basis[a];
                        }

                        //// Derivative of source term ////
                        PetscReal lam;  //lambda damping constant
                        AppCtx *user = (AppCtx *)ctx;
                        lam = user->lam;

                                PetscScalar A = (phi_val*phi_val) - (chi_val*chi_val) + 1.0;
                                PetscScalar C = (phi_val*phi_val) - (chi_val*chi_val) - 1.0;
                                PetscScalar B = C*C + 4.0*phi_val*phi_val;
                        PetscScalar V_phiphi = (-2.0*lam*PetscPowScalar(B, -2.5)) * ((3.0*phi_val*phi_val - chi_val*chi_val + 1.0)*(B) - 6.0*phi_val*phi_val*(A*A));
                        PetscScalar V_phichi = 4.0*lam*phi_val*chi_val*(PetscPowScalar(B, -2.5))*(B - 3.0*A*C);
                        PetscScalar V_chichi = 2.0*lam*((phi_val*phi_val - 3.0*chi_val*chi_val - 1) * PetscPowScalar(B, -1.5) + (6.0*chi_val*chi_val*(C*C))*PetscPowScalar(B, -2.5));
                        PetscScalar V_chiphi = 4.0*lam*phi_val*chi_val*(PetscPowScalar(B, -2.5))*(B - 3.0*C*A);

                        for (int a = 0; a < 8; ++a) {
                                for (int b = 0; b < 8; ++b) {
                                        J_local_pp[a][b] += basis[a] * basis[b] * V_phiphi * weight * detJ;
                                        J_local_pc[a][b] += basis[a] * basis[b] * V_phichi * weight * detJ;
                                        J_local_cc[a][b] += basis[a] * basis[b] * (-1.0)*V_chichi * weight * detJ;
                                        J_local_cp[a][b] += basis[a] * basis[b] * (-1.0)*V_chiphi * weight * detJ;
                                }
                        }
                        }
                }
        }
}
