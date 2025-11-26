#include <stdlib.h>
#include "appctx.h"
#include "nonlin.h"


/*==============================================================*
   Parallel Jacobian for a 2-D DMDA with 4 dofs/point
   – assumes that the DM is stored in user->dm
 *==============================================================*/
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat P, void *ctx)
{

	MatSetOption(J, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
	if (P != J) MatSetOption(P, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
	MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);

	
  AppCtx           *user = (AppCtx*)ctx;
  DM                dm   = user->dm;
  DMDALocalInfo     info;
  MatStencil        row[16], col[16];
  PetscInt          xs,ys,xm,ym, nx,nt;
  PetscReal         hx,ht;
  PetscScalar     (**u)[4];                    /* ghosted view of U – 4 dofs/pt */

  PetscFunctionBegin;
  /* --- scalars from ctx --- */
  nx = user->nx;  nt = user->nt;
  hx = user->hx;  ht = user->ht;
  
  
  /* --- DM layout --- */
  DMDAGetLocalInfo(dm,&info);
  xs = info.xs; xm = info.xm;
  ys = info.ys; ym = info.ym;

  /* --- make sure J & P start clean --- */
  MatZeroEntries(J);          /* works locally, collective later */
  if (P != J) MatZeroEntries(P);

  /* --- element-level constant matrices --- */
	PetscScalar (*A_time)[4]    = user->A_time;
	PetscScalar (*A_space)[4]   = user->A_space;
	PetscScalar (*A_std)[4]     = user->A_standard;
  /* --- read-only access to a ghosted copy of U --- */
  Vec Ul;
  DMGetLocalVector(dm,&Ul);
  DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul);
  DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul);
  DMDAVecGetArrayRead(dm,Ul,&u);

  /* we stop at (nt-2) and have (tElm+1) in the loop */
	PetscInt tEnd = PetscMin(ys + ym - 1, nt - 2);
  /* --- main element loop over *local* rectangles --- */
  for (PetscInt tElm = ys; tElm <=tEnd; ++tElm) {
    for (PetscInt xElm = xs; xElm <xs+xm;   ++xElm) {
		
	PetscInt xloc0 = xElm;
	PetscInt xloc1 = (xElm + 1);        /* may fall in the one‑cell ghost */

	PetscInt xg0   = xElm;             /* for stencils */
	PetscInt xg1   = (xElm + 1);
	PetscInt t0    = tElm;
	PetscInt t1    = tElm + 1;

	PetscInt xglob[4] = { xg0, xg1, xg1, xg0 };
	PetscInt tglob[4] = {  t0,  t0,  t1,  t1 };
	int k = 0;
	for (int corner = 0; corner < 4; ++corner) {
	  for (int comp = 0; comp < 4; ++comp, ++k) {
		row[k].i = xglob[corner];
		row[k].j = tglob[corner];
		row[k].c = comp;
		col[k]   = row[k];
	  }
	} 

	
      /* -------- pull the element’s phi,chi values from ghosted U -------- */
      PetscScalar phi_l[4], chi_l[4];
	  
    phi_l[0]=u[t0][xloc0][0];  
	  phi_l[1]=u[t0][xloc1][0];
    phi_l[2]=u[t1][xloc1][0];  
	  phi_l[3]=u[t1][xloc0][0];

    chi_l[0]=u[t0][xloc0][2];  
	  chi_l[1]=u[t0][xloc1][2];
    chi_l[2]=u[t1][xloc1][2];  
	  chi_l[3]=u[t1][xloc0][2];

      /* -------- build 16×16 element matrix M_el -------- */
      PetscScalar M[16][16] = {{0}};
      for (int i=0;i<4;++i){
        int r1 = 4*i, r2=r1+1, r3=r1+2, r4=r1+3;
        for (int j=0;j<4;++j){
          int cphi = 4*j,  cu=cphi+1,  cchi=cphi+2,  cv=cphi+3;
          /* linear part identical to your serial code */
          M[r1][cphi] += A_space[i][j] + user->mphi2*A_std[i][j];
          M[r1][cu]   += A_time [i][j];
          M[r2][cphi] += A_time [i][j];
          M[r2][cu]   += -A_std [i][j];

          M[r3][cchi] += A_space[i][j] + user->mchi2*A_std[i][j];
          M[r3][cv]   += A_time [i][j];
          M[r4][cchi] += A_time [i][j];
          M[r4][cv]   += -A_std [i][j];
        }
      }

      /* -------- nonlinear 2×2 block coupling phi,chi -------- */
      PetscScalar Jpp[4][4], Jpc[4][4], Jcc[4][4], Jcp[4][4];
      ComputeJ_local_nonlinear(Jpp,Jpc,Jcc,Jcp,phi_l,chi_l,hx,ht,ctx);
      for (int i=0;i<4;++i){
        int r1=4*i, r3=r1+2;
        for (int j=0;j<4;++j){
          int cphi=4*j, cchi=cphi+2;
          M[r1][cphi]+=Jpp[i][j];  M[r1][cchi]+=Jpc[i][j];
          M[r3][cchi]+=Jcc[i][j];  M[r3][cphi]+=Jcp[i][j];
        }
      }

      /* -------- assemble – local indices, no duplicates -------- */

	MatSetValuesStencil(J,16,row,16,col,&M[0][0],ADD_VALUES);
	if (P!=J) MatSetValuesStencil(P,16,row,16,col,&M[0][0],ADD_VALUES);
	
	
	}
  }

  /* --- restore --- */
  DMDAVecRestoreArrayRead(dm,Ul,&u);
  DMRestoreLocalVector(dm,&Ul);

  /* --- first global assembly pass --- */
  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  if (P!=J){MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);}

  /* ------------------------------------------------------------------ *
     Impose t=0 initial-condition rows.
     Only the rank(s) that own t=0 have ys==0, so we do it locally then
     call MatZeroRowsLocal() .
   *------------------------------------------------------------------ */
 
	/* build a single array of all the t=0 rows on this rank */
	PetscInt    nlocal   = 0;
	MatStencil *bc       = NULL;
		
	if (info.ys == 0){
		nlocal = info.xm * 4;                    /* 4 dofs per x‑node */         
		PetscMalloc1(nlocal,&bc);
		
		PetscInt    idx    = 0;
		for (PetscInt x=info.xs; x<info.xs+info.xm; ++x) {
		  for (int comp=0; comp<4; ++comp, ++idx) {
			bc[idx].i = x;        /* spatial index */
			bc[idx].j = 0;        /* t = 0 layer */ 
			bc[idx].c = comp;     /* component */
		  }
		}
	}
	
	/* zero those rows on every rank (others will just ignore them) */
	MatZeroRowsStencil(J, nlocal, bc, 1.0, NULL, NULL);
	if (P != J) MatZeroRowsStencil(P, nlocal, bc, 1.0, NULL, NULL);
	
	if (bc) PetscFree(bc);
	
	  /* --- final assembly after row-zeroing --- */
	MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
	if (P!=J){MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);}


  PetscFunctionReturn(0);
}