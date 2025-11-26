#include <stdlib.h>
#include "appctx.h"
#include "nonlin.h"
#include "ic.h"

PetscErrorCode FormResidual(SNES snes, Vec U, Vec R, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  DM                 dm   = user->dm;          
  DMDALocalInfo      info;
  PetscInt           xs,ys,xm,ym,nx,nt;
  PetscReal          hx,ht,xL,xR;

  PetscFunctionBegin;
  

  /* ---------- scalars that were already in ctx ---------- */
  nx   = user->nx;
  nt   = user->nt;
  hx   = user->hx;
  ht   = user->ht;
  xL   = user->xL;

  
	/* --- element-level constant matrices --- */
	PetscScalar (*A_time)[4]    = user->A_time;
	PetscScalar (*A_space)[4]   = user->A_space;
  PetscScalar (*A_standard)[4]= user->A_standard;

  /* ---------- DMDA locality ---------- */
  DMDAGetLocalInfo(dm,&info);          /* fills info.mx,my,… xs,xm,…            */
  xs = info.xs;  xm = info.xm;         /* x-start and width that this rank owns */
  ys = info.ys;  ym = info.ym;         /* t-start and height                     */

  /* ---------- zero the residual ---------- */	

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
		

	
	/* --- array access is now to the local vectors --- */
	PetscScalar (**u)[2], (**r)[2];
	DMDAVecGetArrayRead(dm,Ul,&u);
	DMDAVecGetArray    (dm,Rl,&r);
	
 

  /* we stop at (nt-2) and (nx - 2) and have (tElm+1) and (xElm+1) in the loop */
  PetscInt tEnd = PetscMin(ys + ym - 1, nt - 2);
  PetscInt xEnd = PetscMin(xs + xm - 1, nx - 2);
  /* ---------- element-wise loop – LOCAL region only ---- */
  for (PetscInt tElm = ys; tElm <=tEnd; ++tElm) {            /* nt-1 elements in t */
    for (PetscInt xElm = xs; xElm <=xEnd;   ++xElm) {          /* nx-1 elements in x  */
		
      /* ---- grab the four corner values from ghosted array ---- */
      PetscScalar u_local[4], v_local[4];
	  
		PetscInt xloc1 = (xElm + 1);
		
		u_local[0] = u[tElm  ][xElm ][0];
		u_local[1] = u[tElm  ][xloc1][0];
		u_local[2] = u[tElm+1][xloc1][0];
		u_local[3] = u[tElm+1][xElm ][0];

		v_local[0] = u[tElm  ][xElm ][1];
		v_local[1] = u[tElm  ][xloc1][1];
		v_local[2] = u[tElm+1][xloc1][1];
		v_local[3] = u[tElm+1][xElm ][1];
		


      /* ---- build local residuals ---- */
      PetscScalar eqn1_r_local[4]={0}; /* 2 eqn in wave weak form */
      PetscScalar eqn2_r_local[4]={0};

      for (PetscInt i=0;i<4;++i){
        for (PetscInt j=0;j<4;++j){
          eqn1_r_local[i] += A_time[i][j]*v_local [j] + A_space[i][j]*u_local[j];
          eqn2_r_local[i] += -A_time[i][j]*u_local [j] + A_standard[i][j]*v_local[j];
        }
      }

      
      PetscScalar eqn1_r_nonlin[4];
	  /* This problem is linear, and we use this to fill the source term. 
    We keep the naming convention for future use */
    PetscReal x0 = xL + xElm*hx;
		PetscReal t0 = tElm*ht;	  
      ComputeR_local_nonlinear(eqn1_r_nonlin,u_local,hx,ht,x0,t0,ctx);
      for (PetscInt i=0;i<4;++i){
        eqn1_r_local[i]+=eqn1_r_nonlin[i];
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

    }
  }

	VecAssemblyBegin(Rl);       
	VecAssemblyEnd  (Rl);

	DMDAVecRestoreArrayRead(dm,Ul,&u);
	DMDAVecRestoreArray    (dm,Rl,&r);
	DMLocalToGlobalBegin(dm,Rl,ADD_VALUES,R);
	DMLocalToGlobalEnd  (dm,Rl,ADD_VALUES,R);
	
  PetscScalar (**rg)[2];                        /* work directly in global R  */
  PetscScalar (**ug)[2];                        /* Ul is still valid and ghosted*/
  DMDAVecGetArray    (dm,R,&rg);
  DMDAVecGetArrayRead(dm,Ul,&ug);
  

  /* t=0 initial condition */
  if (ys==0){
    if(user->IC == 1)
    {
      for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {
        PetscReal x_phys = xL + x*hx;
        rg[0][x][0] = ug[0][x][0] - wave_IC_u(x_phys);
        rg[0][x][1] = ug[0][x][1] - wave_IC_v(x_phys);
      }
    }	
  }

  DMDAVecRestoreArrayRead(dm,Ul,&ug);
	DMDAVecRestoreArray    (dm,R,&rg);

	VecAssemblyBegin(R);  VecAssemblyEnd(R);
	DMRestoreLocalVector(dm,&Ul);
	DMRestoreLocalVector(dm,&Rl);

  PetscFunctionReturn(0);
}