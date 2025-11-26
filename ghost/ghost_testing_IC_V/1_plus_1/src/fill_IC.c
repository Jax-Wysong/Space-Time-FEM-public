#include "fill_IC.h"
#include "ic.h"
#include "appctx.h"


PetscErrorCode FillInitialConditions(DM dm, Vec U, AppCtx *user)
{
  PetscFunctionBeginUser;
  DMDALocalInfo info;
  PetscScalar (**u)[4];
  PetscInt xs,xm,ys,ym;

  PetscCall(DMDAGetLocalInfo(dm, &info));
  xs = info.xs; xm = info.xm;
  ys = info.ys; ym = info.ym;

  PetscCall(DMDAVecGetArray(dm,U,&u));
  for (PetscInt t = ys; t < ys+ym; ++t) {
    for (PetscInt x = xs; x < xs+xm; ++x) {
      PetscReal x_phys = user->xL + x * user->hx;
      if (t == 0) {
			if(user->IC == 1)
			{
				u[t][x][0] = phi_wave_IC(x_phys,user->A,user->C);
				u[t][x][1] = u_wave_IC  (x_phys,user->mphi2,user->A,user->C);
				u[t][x][2] = chi_wave_IC(x_phys,user->A,user->C);
				u[t][x][3] = v_wave_IC  (x_phys,user->mchi2,user->A,user->C);
			}
			if(user->IC == 2)
			{
				u[t][x][0] = phi_gauss_IC(x_phys,user->A,user->C);
				u[t][x][1] = u_gauss_IC  (x_phys,user->mphi2,user->A,user->C);
				u[t][x][2] = chi_gauss_IC(x_phys,user->A,user->C);
				u[t][x][3] = v_gauss_IC  (x_phys,user->mchi2,user->A,user->C);
			}
			if(user->IC == 3)
			{
				u[t][x][0] = phi_noise_IC(x_phys, user->A, user->ns,
										  user->n1,user->n2,user->L, user->theta);
				u[t][x][1] = u_noise_IC  (x_phys, user->mphi2, user->A, user->ns,
										  user->n1,user->n2,user->L, user->theta, user->ssign);
				u[t][x][2] = chi_noise_IC(x_phys, user->A, user->ns,
										  user->n1,user->n2,user->L, user->theta, user->dphi);
				u[t][x][3] = v_noise_IC  (x_phys, user->mchi2, user->A, user->ns,
										  user->n1,user->n2,user->L, user->theta, user->ssign, user->dphi);
			}
			if(user->IC == 4)
			{
				u[t][x][0] = phi_pw_IC(x_phys, user->A);
				u[t][x][1] = u_pw_IC  (x_phys, user->mphi2, user->A);
				u[t][x][2] = chi_pw_IC(x_phys, user->A, user->pw_r,user->pw_dphi);
				u[t][x][3] = v_pw_IC  (x_phys, user->mchi2, user->A, user->pw_r, user->pw_dphi, user->pw_sigma);
			}
			if (user->IC == 5) {
			  u[t][x][0] = phi_osc_IC(x_phys, user->A,
									  user->osc_x0, user->osc_sigma, user->osc_k0);
			  u[t][x][1] = u_osc_IC  (x_phys, user->mphi2, user->A,
									  user->osc_x0, user->osc_sigma, user->osc_k0);
			  u[t][x][2] = chi_osc_IC(x_phys, user->A, user->osc_r,
									  user->osc_x0, user->osc_sigma, user->osc_dphi, user->osc_k0);
			  u[t][x][3] = v_osc_IC  (x_phys, user->mchi2, user->A, user->osc_r,
									  user->osc_x0, user->osc_sigma, user->osc_dphi, user->osc_k0);
			}
      } else {
        u[t][x][0] = u[t][x][1] = u[t][x][2] = u[t][x][3] = 0.0;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(dm,U,&u));
  PetscFunctionReturn(0);
}
