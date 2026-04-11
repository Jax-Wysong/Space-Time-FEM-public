#include <stdlib.h>
#include "appctx.h"
#include "nonlin.h"
#include "ic.h"

PetscErrorCode FormResidual(SNES snes, Vec U, Vec R, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  DM                 dm   = user->dm;          
  DMDALocalInfo      info;
  PetscInt           xs,ys,xm,ym,nx,nt,IC,n1,n2,pw_sigma;
  PetscReal          hx,ht,xL,xR,mphi2,mchi2,A, mass_phi2, mass_chi2,ns,L,dphi,pw_r,pw_dphi,C;

  PetscFunctionBegin;
  

  /* ---------- scalars that were already in ctx ---------- */
  nx   = user->nx;
  nt   = user->nt;
  hx   = user->hx;
  ht   = user->ht;
  xL   = user->xL;
  mphi2 = user->mphi2;
  mchi2 = user->mchi2;
  A    = user->A;
  IC 	= user->IC;
  mass_phi2  = mphi2;
  mass_chi2  = mchi2;
  
	/* --- element-level constant matrices --- */
	PetscScalar (*A_time)[4]    = user->A_time;
	PetscScalar (*A_space)[4]   = user->A_space;
	PetscScalar (*A_standard)[4]     = user->A_standard;

  /* ---------- DMDA locality ---------- */
  DMDAGetLocalInfo(dm,&info);          /* fills info.mx,my,… xs,xm,…            */
  xs = info.xs;  xm = info.xm;         /* x-start and width that THIS rank owns */
  ys = info.ys;  ym = info.ym;         /* t-start and height                     */

  /* ---------- zero the residual ---------- */

	VecAssemblyBegin(R);
	VecAssemblyEnd  (R);

  VecZeroEntries(R);                  

	/* --- get ghosted work vectors --- */
	Vec Ul, Rl;
	DMGetLocalVector   (dm,&Ul);
	DMGetLocalVector   (dm,&Rl);
	
	PetscScalar *a; PetscInt nloc;
	VecGetLocalSize(Rl,&nloc);
	VecGetArrayWrite(Rl,&a);  PetscArrayzero(a,nloc);  VecRestoreArrayWrite(Rl,&a);
		

	/* --- fill Ul with ghosts, zero Rl --- */
	DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul);
	DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul);
		

	
	/* --- array access is now to the *local* vectors --- */
	PetscScalar (**u)[4], (**r)[4];
	DMDAVecGetArrayRead(dm,Ul,&u);
	DMDAVecGetArray    (dm,Rl,&r);
	
 


  /* we stop at (nt-2) and have (tElm+1) in the loop */
  PetscInt tEnd = PetscMin(ys + ym - 1, nt - 2);
  /* ---------- element-wise loop – LOCAL region only ---- */
  for (PetscInt tElm = ys; tElm <=tEnd; ++tElm) {            /* nt-1 elements in t */
    for (PetscInt xElm = xs; xElm <xs+xm;   ++xElm) {          /* all owned x cols (periodic, dmda handles) */
		
      /* ---- grab the four corner values from ghosted array ---- */
      PetscScalar phi_local[4], u_local[4], chi_local[4], v_local[4];
	  
		PetscInt xloc1 = (xElm + 1);
		

		phi_local[0] = u[tElm  ][xElm ][0];
		phi_local[1] = u[tElm  ][xloc1][0];
		phi_local[2] = u[tElm+1][xloc1][0];
		phi_local[3] = u[tElm+1][xElm ][0];
		
		u_local[0] = u[tElm  ][xElm ][1];
		u_local[1] = u[tElm  ][xloc1][1];
		u_local[2] = u[tElm+1][xloc1][1];
		u_local[3] = u[tElm+1][xElm ][1];
		
		chi_local[0] = u[tElm  ][xElm ][2];
		chi_local[1] = u[tElm  ][xloc1][2];
		chi_local[2] = u[tElm+1][xloc1][2];
		chi_local[3] = u[tElm+1][xElm ][2];
		
		v_local[0] = u[tElm  ][xElm ][3];
		v_local[1] = u[tElm  ][xloc1][3];
		v_local[2] = u[tElm+1][xloc1][3];
		v_local[3] = u[tElm+1][xElm ][3];


      /* ---- build local residuals  ---- */
      PetscScalar eqn1_r_local[4]={0}, eqn2_r_local[4]={0};
      PetscScalar eqn3_r_local[4]={0}, eqn4_r_local[4]={0};

      for (PetscInt i=0;i<4;++i){
        for (PetscInt j=0;j<4;++j){
          eqn1_r_local[i] += A_time[i][j]*u_local [j] + A_space[i][j]*phi_local[j] + mass_phi2 * A_standard[i][j]*phi_local[j];
          eqn2_r_local[i] += A_time[i][j]*phi_local[j] - A_standard[i][j]*u_local[j];

          eqn3_r_local[i] += A_time[i][j]*v_local [j] + A_space[i][j]*chi_local[j] + mass_chi2 * A_standard[i][j]*chi_local[j];
          eqn4_r_local[i] += A_time[i][j]*chi_local[j] - A_standard[i][j]*v_local[j];
        }
      }


      /* get the nonlinear contributions from the potential */
      PetscScalar eqn1_r_nonlin[4], eqn3_r_nonlin[4];
      PetscReal x0 = xL + xElm*hx;
      PetscReal t0 = tElm*ht;	  
      ComputeR_local_nonlinear(eqn1_r_nonlin,eqn3_r_nonlin,phi_local,chi_local,hx,ht,x0,t0,ctx);
      for (PetscInt i=0;i<4;++i){
        eqn1_r_local[i]+=eqn1_r_nonlin[i];
        eqn3_r_local[i]+=eqn3_r_nonlin[i];
      }

      /* ---- scatter into residual vector ---- */
	  
      r[tElm  ][xElm ][0] += eqn1_r_local[0];
      r[tElm  ][xloc1][0] += eqn1_r_local[1];
      r[tElm+1][xloc1][0] += eqn1_r_local[2];
      r[tElm+1][xElm ][0] += eqn1_r_local[3];
	  
      r[tElm  ][xElm ][1] += eqn2_r_local[0];
      r[tElm  ][xloc1][1] += eqn2_r_local[1];
      r[tElm+1][xloc1][1] += eqn2_r_local[2];
      r[tElm+1][xElm ][1] += eqn2_r_local[3];

      r[tElm  ][xElm ][2] += eqn3_r_local[0];
      r[tElm  ][xloc1][2] += eqn3_r_local[1];
      r[tElm+1][xloc1][2] += eqn3_r_local[2];
      r[tElm+1][xElm ][2] += eqn3_r_local[3];
	  
      r[tElm  ][xElm ][3] += eqn4_r_local[0];
      r[tElm  ][xloc1][3] += eqn4_r_local[1];
      r[tElm+1][xloc1][3] += eqn4_r_local[2];
      r[tElm+1][xElm ][3] += eqn4_r_local[3];
	  

    }
  }

	VecAssemblyBegin(Rl);       
	VecAssemblyEnd  (Rl);

	DMDAVecRestoreArrayRead(dm,Ul,&u);
	DMDAVecRestoreArray    (dm,Rl,&r);
	DMLocalToGlobalBegin(dm,Rl,ADD_VALUES,R);
	DMLocalToGlobalEnd  (dm,Rl,ADD_VALUES,R);
	


  /* t=0 initial condition */
  if (ys==0){
    PetscScalar (**rg)[4];                        /* work directly in global R  */
	PetscScalar (**ug)[4];                        /* Ul is still valid & ghosted*/
	DMDAVecGetArray    (dm,R,&rg);
	DMDAVecGetArrayRead(dm,Ul,&ug);

	if(IC == 1)
	{
		for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {
			PetscReal x_phys = xL + x*hx;
			rg[0][x][0] = ug[0][x][0] - phi_MMS_IC(x_phys);
			rg[0][x][1] = ug[0][x][1] - u_MMS_IC  (x_phys);
			rg[0][x][2] = ug[0][x][2] - chi_MMS_IC(x_phys);
			rg[0][x][3] = ug[0][x][3] - v_MMS_IC  (x_phys);
		}
	}
	
	DMDAVecRestoreArrayRead(dm,Ul,&ug);
	DMDAVecRestoreArray    (dm,R,&rg);

	
  }


	VecAssemblyBegin(R);  VecAssemblyEnd(R);
	DMRestoreLocalVector(dm,&Ul);
	DMRestoreLocalVector(dm,&Rl);

  PetscFunctionReturn(0);
}