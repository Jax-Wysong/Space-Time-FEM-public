#include <stdlib.h>
#include "appctx.h"
#include "ic.h"
#include "source.h"

PetscErrorCode FormResidual(SNES snes, Vec U, Vec R, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  DM                 dm   = user->dm;          
  DMDALocalInfo      info;
  PetscInt           xs,ys,xm,ym,nx,nt;
  PetscReal          hx,ht,xL,xR;

  PetscFunctionBegin;
  

  /* ---------- scalars that are already in ctx ---------- */
  nx   = user->nx;
  nt   = user->nt;
  hx   = user->hx;
  ht   = user->ht;
  xL   = user->xL;

  
	/* --- element-level constant matrices --- */
	PetscScalar (*A_time)[4]    = user->A_time;
	PetscScalar (*A_space)[4]   = user->A_space;
  PetscScalar (*A_standard)[4]= user->A_standard;
  PetscScalar (*A_time_edge_R)[4] = user->A_time_edge_R; 
  PetscScalar (*A_time_edge_L)[4] = user->A_time_edge_L;

  /* ------ Potential terms ------- */
  PetscScalar *Vx = user->V_of_x;

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
		

	/* --- fill Ul with ghosts --- */
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
      PetscScalar eqn1_r_local[4]={0}; /* 2 eqn in weak form */
      PetscScalar eqn2_r_local[4]={0};

      /* ---- element potential value ---- */
      PetscScalar V_left  = Vx[xElm];
      PetscScalar V_right = Vx[xElm+1];
      PetscScalar V_elem  = 0.5 * (V_left + V_right); // average


      for (PetscInt i=0;i<4;++i){
        for (PetscInt j=0;j<4;++j){
          /* eqn 1: v_t - u_xx + Vu */
          eqn1_r_local[i] += A_time[i][j]*v_local [j] + A_space[i][j]*u_local[j] + A_standard[i][j]*u_local[j]*V_elem;

          /* eqn 2: -u_t + v */
          eqn2_r_local[i] += -A_time[i][j]*u_local [j] + A_standard[i][j]*v_local[j];
        }
      }

    /* Include boundary contributions*/
    /* left boundary: -int v w dt at x_L  */
    if (xElm == 0) {
        for (PetscInt i=0;i<4;++i)
          for (PetscInt j=0;j<4;++j)
            eqn1_r_local[i] -= A_time_edge_L[i][j] * v_local[j];
    }
    /* right boundary: -int v w dt at x_R */
    if (xElm == nx-2) {
        for (PetscInt i=0;i<4;++i)
          for (PetscInt j=0;j<4;++j)
            eqn1_r_local[i] -= A_time_edge_R[i][j] * v_local[j];
    }


      
      PetscScalar eqn1_r_pp[4];
	  /* Fill in the particle source term information */
    PetscReal x0 = xL + xElm*hx;
		PetscReal t0 = tElm*ht;	  
      ComputeR_local_pp(eqn1_r_pp,hx,ht,x0,t0,ctx);
      for (PetscInt i=0;i<4;++i){
        /* already gets a minus sign upon creation */
        eqn1_r_local[i]+=eqn1_r_pp[i];
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
  PetscScalar (**ug)[2];                        /* Ul is still valid & ghosted*/
  DMDAVecGetArray    (dm,R,&rg);
  DMDAVecGetArrayRead(dm,Ul,&ug);
  

  /* t=0 initial condition */
  if (ys==0){
    if(user->IC == 1)
    {
      for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {
        PetscReal x_phys = xL + x*hx;
        rg[0][x][0] = ug[0][x][0] - BH_IC_u(x_phys); /* both of these are 0.0 because we are */
        rg[0][x][1] = ug[0][x][1] - BH_IC_v(x_phys); /* interested in seeing what the source term does*/
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