#include <stdlib.h>
#include "appctx.h"
// #include "nonlin.h"

/*==============================================================*
   Parallel Jacobian for a 2-D DMDA with 1 dofs/point
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
  MatStencil        row[4], col[4];
  PetscInt          xs,ys,xm,ym, nx,nt;
  PetscReal         hx,ht;
  PetscScalar     (**u)[1];                    /* ghosted view of U ; 1 dofs/pt */

  PetscFunctionBegin;
  /* --- scalars from ctx --- */
  nx = user->nx;  nt = user->nt;
  hx = user->hx;  ht = user->ht;
  
  
  /* --- DM layout --- */
  DMDAGetLocalInfo(dm,&info);
  xs = info.xs; xm = info.xm;
  ys = info.ys; ym = info.ym;

  /* --- make sure J & P start clean --- */
  MatZeroEntries(J);          
  if (P != J) MatZeroEntries(P);

  /* --- element-level constant matrices --- */
	PetscScalar (*A_time)[4]    = user->A_time;
	PetscScalar (*A_space)[4]   = user->A_space;

  /* --- Read-only access to a ghosted copy of U --- */
  Vec Ul;
  DMGetLocalVector(dm,&Ul);
  DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul);
  DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul);
  DMDAVecGetArrayRead(dm,Ul,&u);

  /* we stop at (nt-2) and (nx - 2) and have (tElm+1) and (xElm+1) in the loop */
	PetscInt tEnd = PetscMin(ys + ym - 1, nt - 2);
  PetscInt xEnd = PetscMin(xs + xm - 1, nx - 2);
  /* --- main element loop over local rectangles --- */
  for (PetscInt tElm = ys; tElm <=tEnd; ++tElm) {
    for (PetscInt xElm = xs; xElm <=xEnd;   ++xElm) {
		
    /* for getting values from U if needed*/
	PetscInt xloc0 = xElm;
	PetscInt xloc1 = (xElm + 1); 

  /* for stencils */      
	PetscInt xg0   = xElm;            
	PetscInt xg1   = (xElm + 1);
	PetscInt t0    = tElm;
	PetscInt t1    = tElm + 1;
	PetscInt xglob[4] = { xg0, xg1, xg1, xg0 };
	PetscInt tglob[4] = {  t0,  t0,  t1,  t1 };
	int k = 0;
	for (int corner = 0; corner < 4; ++corner) {
	  for (int comp = 0; comp < 1; ++comp, ++k) {
		row[k].i = xglob[corner];
		row[k].j = tglob[corner];
		row[k].c = comp;
		col[k]   = row[k];
	  }
	} 

	
      /* -------- pull the element’s phi,chi values from ghosted U -------- */
      PetscScalar u_l[4];
	  
    u_l[0]=u[t0][xloc0][0];  
	  u_l[1]=u[t0][xloc1][0];
    u_l[2]=u[t1][xloc1][0];  
	  u_l[3]=u[t1][xloc0][0];


      /* -------- build 4×4 element matrix -------- */
      PetscScalar M[4][4] = {{0}};
      for (int i=0;i<4;++i){
        int r1 = 1*i;
        for (int j=0;j<4;++j){
          int cu = 1*j;
          /* linear part  */
          M[r1][cu] += A_time [i][j] + A_space[i][j];
        }
      }
      /* -------- assemble – local indices -------- */

	MatSetValuesStencil(J,4,row,4,col,&M[0][0],ADD_VALUES);
	if (P!=J) MatSetValuesStencil(P,4,row,4,col,&M[0][0],ADD_VALUES);
	
	
	}
  }

  /* --- restore --- */
  DMDAVecRestoreArrayRead(dm,Ul,&u);
  DMRestoreLocalVector(dm,&Ul);

  /* --- first global assembly pass --- */
  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  if (P!=J){MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);}

  /* ------------------------------------------------------------------ *
     Impose t=0 initial-condition rows
     and 0 out the boundaries for heat BC
     Only the rank(s) that own t=0 have ys==0, so we do it locally then
     call MatZeroRowsLocal() (no global broadcast needed).
   *------------------------------------------------------------------ */
 
    PetscInt    nlocal = 0;
    MatStencil *bc     = NULL;

    /* --- First, count how many BC rows we need on this rank --- */
    PetscInt count = 0;

    /* t = 0 initial condition rows */
    if (info.ys == 0) {
      count += info.xm * 1;          /* 1 dof per x-node */
    }

    /* x = 0 boundary rows (all local t) */
    if (info.xs == 0) {
      count += info.ym * 1;          /* 1 dof per t-node */
    }

    /* x = nx-1 boundary rows (all local t) */
    if (info.xs + info.xm == user->nx) {
      count += info.ym * 1;
    }

    /* If this rank has any BC rows at all, allocate and fill */
    if (count > 0) {
      PetscMalloc1(count, &bc);

      PetscInt idx = 0;

      /* --- t = 0 rows: all local x at t=0 --- */
      if (info.ys == 0) {
        for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {
          bc[idx].i = x;     /* spatial index */
          bc[idx].j = 0;     /* t = 0 */
          bc[idx].c = 0;     /* component */
          ++idx;
        }
      }

      /* --- x = 0 rows: all local t at x=0 --- */
      if (info.xs == 0) {
        for (PetscInt t = info.ys; t < info.ys + info.ym; ++t) {
          bc[idx].i = 0;     /* x = 0 */
          bc[idx].j = t;     /* time index */
          bc[idx].c = 0;
          ++idx;
        }
      }

      /* --- x = nx-1 rows: all local t at x = nx-1 --- */
      if (info.xs + info.xm == user->nx) {
        PetscInt x = user->nx - 1;
        for (PetscInt t = info.ys; t < info.ys + info.ym; ++t) {
          bc[idx].i = x;     /* x = nx-1 */
          bc[idx].j = t;
          bc[idx].c = 0;
          ++idx;
        }
      }

      nlocal = idx;  /* total # of stencil rows on this rank */
    }

    /* Zero those rows & put 1 on the diagonal */
    MatZeroRowsStencil(J, nlocal, bc, 1.0, NULL, NULL);
    if (P != J) MatZeroRowsStencil(P, nlocal, bc, 1.0, NULL, NULL);

    PetscFree(bc);

    /* final assembly after row-zeroing */
    MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
    if (P != J) {
      MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);
    }

  PetscFunctionReturn(0);
}