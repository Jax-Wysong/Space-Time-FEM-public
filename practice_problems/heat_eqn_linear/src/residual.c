#include <stdlib.h>
#include "appctx.h"
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

  /* ---------- DMDA locality ---------- */
  DMDAGetLocalInfo(dm,&info);          /* fills info.mx,my,… xs,xm,…            */
  xs = info.xs;  xm = info.xm;         /* x-start and width that THIS rank owns */
  ys = info.ys;  ym = info.ym;         /* t-start and height                     */

  /* ---------- zero the residual the PETSc way ---------- */	

  VecZeroEntries(R);                   /* touches only local part under the hood*/

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
	PetscScalar (**u)[1], (**r)[1];
	DMDAVecGetArrayRead(dm,Ul,&u);
	DMDAVecGetArray    (dm,Rl,&r);
	
 


  /* we stop at (nt-2) and (nx - 2) and have (tElm+1) and (xElm+1) in the loop */
  PetscInt tEnd = PetscMin(ys + ym - 1, nt - 2);
  PetscInt xEnd = PetscMin(xs + xm - 1, nx - 2);
  /* ---------- element-wise loop – LOCAL region only ---- */
  for (PetscInt tElm = ys; tElm <=tEnd; ++tElm) {            /* nt-1 elements in t */
    for (PetscInt xElm = xs; xElm <=xEnd;   ++xElm) {          /* all owned x cols   */
		
      /* ---- grab the four corner values from ghosted array ---- */
      PetscScalar u_local[4];
	  
		PetscInt xloc1 = (xElm + 1);
		
		u_local[0] = u[tElm  ][xElm ][0];
		u_local[1] = u[tElm  ][xloc1][0];
		u_local[2] = u[tElm+1][xloc1][0];
		u_local[3] = u[tElm+1][xElm ][0];
		


      /* ---- build local residuals ---- */
      PetscScalar eqn1_r_local[4]={0}; /* only 1 eqn in heat weak form */
      

      for (PetscInt i=0;i<4;++i){
        for (PetscInt j=0;j<4;++j){
          eqn1_r_local[i] += A_time[i][j]*u_local [j] + A_space[i][j]*u_local[j];
        }
      }

      /* ---- scatter into residual vector ---- */
	  
      r[tElm  ][xElm ][0] += eqn1_r_local[0];
      r[tElm  ][xloc1][0] += eqn1_r_local[1];
      r[tElm+1][xloc1][0] += eqn1_r_local[2];
      r[tElm+1][xElm ][0] += eqn1_r_local[3];	  

    }
  }

	VecAssemblyBegin(Rl);       
	VecAssemblyEnd  (Rl);

	DMDAVecRestoreArrayRead(dm,Ul,&u);
	DMDAVecRestoreArray    (dm,Rl,&r);
	DMLocalToGlobalBegin(dm,Rl,ADD_VALUES,R);
	DMLocalToGlobalEnd  (dm,Rl,ADD_VALUES,R);
	
  PetscScalar (**rg)[1];                        /* work directly in global R  */
  PetscScalar (**ug)[1];                        /* Ul is still valid & ghosted*/
  DMDAVecGetArray    (dm,R,&rg);
  DMDAVecGetArrayRead(dm,Ul,&ug);
  

  /* t=0 initial condition */
  if (ys==0){
    if(user->IC == 1)
    {
      for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {
        PetscReal x_phys = xL + x*hx;
        rg[0][x][0] = ug[0][x][0] - heat_linear_IC(x_phys);
      }
    }	
  }

  /* boundary conditions need us to zero them out */
  for (PetscInt t = info.ys; t < info.ys + info.ym; ++t) {
    if (info.xs == 0) {                /* left boundary owned on this rank */
      PetscInt x = 0;
      rg[t][x][0] = ug[t][x][0] - 0.0; /* u(t,0) = 0 */
    }
    if (info.xs + info.xm == user->nx) { /* right boundary owned here */
      PetscInt x = user->nx - 1;
      rg[t][x][0] = ug[t][x][0] - 0.0;   /* u(t,L) = 0 */
    }
  } 


  DMDAVecRestoreArrayRead(dm,Ul,&ug);
	DMDAVecRestoreArray    (dm,R,&rg);

	VecAssemblyBegin(R);  VecAssemblyEnd(R);
	DMRestoreLocalVector(dm,&Ul);
	DMRestoreLocalVector(dm,&Rl);

  PetscFunctionReturn(0);
}