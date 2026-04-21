#include <stdlib.h>
#include "appctx.h"
#include "nonlin.h"

/*==============================================================*
   Parallel Jacobian for a 2-D DMDA with 4 dofs/point
   – assumes that the DM is stored in user->dm
 *==============================================================*/
PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat P, void *ctx)
{

	// MatSetOption(J, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
	// if (P != J) MatSetOption(P, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
	// MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);

	
  AppCtx           *user = (AppCtx*)ctx;
  DM                dm   = user->dm;
  DMDALocalInfo     info;
  MatStencil        row[8], col[8];
  PetscInt          xs,ys,xm,ym, nx,nt;
  PetscReal         hx,ht;
  PetscScalar     (**u)[2];                    /* ghosted view of U – 4 dofs/pt */

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
	PetscScalar (*A_standard)[4]     = user->A_standard;



  /* --- read-only access to a ghosted copy of U --- */
  Vec Ul;
  DMGetLocalVector(dm,&Ul);
  DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul);
  DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul);
  DMDAVecGetArrayRead(dm,Ul,&u);


  /* we stop at (nt-2) and (nx - 2) and have (tElm+1) and (xElm+1) in the loop */
	PetscInt tEnd = PetscMin(ys + ym - 1, nt - 2);
  PetscInt xEnd = PetscMin(xs + xm - 1, nx - 2);
  /* --- main element loop over *local* rectangles --- */
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
      for (int comp = 0; comp < 2; ++comp, ++k) {
      row[k].i = xglob[corner];
      row[k].j = tglob[corner];
      row[k].k = 0;
      row[k].c = comp;
      col[k]   = row[k];
      }
    } 

    for (int q=0; q<8; ++q) {
      PetscCheck(row[q].i >= 0 && row[q].i < nx, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "row[%d].i=%D out of [0,%D)", q, row[q].i, nx);
      PetscCheck(row[q].j >= 0 && row[q].j < nt, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "row[%d].j=%D out of [0,%D)", q, row[q].j, nt);
      PetscCheck(row[q].c >= 0 && row[q].c < 2,  PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "row[%d].c=%D out of [0,2)", q, row[q].c);
      PetscCheck(row[q].k == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "row[%d].k=%D expected 0", q, row[q].k);
    }

	
      /* -------- pull the element’s u values from ghosted U -------- */

      PetscInt gx0 = info.gxs, gx1 = info.gxs + info.gxm;
      PetscInt gy0 = info.gys, gy1 = info.gys + info.gym;

      PetscCheck(xloc0 >= gx0 && xloc0 < gx1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "xloc0=%D not in ghosted [%D,%D) (xs=%D xm=%D gxs=%D gxm=%D)", xloc0,gx0,gx1, info.xs,info.xm, info.gxs,info.gxm);
      PetscCheck(xloc1 >= gx0 && xloc1 < gx1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "xloc1=%D not in ghosted [%D,%D) (xs=%D xm=%D gxs=%D gxm=%D)", xloc1,gx0,gx1, info.xs,info.xm, info.gxs,info.gxm);
      PetscCheck(t0 >= gy0 && t0 < gy1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "t0=%D not in ghosted [%D,%D) (ys=%D ym=%D gys=%D gym=%D)", t0,gy0,gy1, info.ys,info.ym, info.gys,info.gym);
      PetscCheck(t1 >= gy0 && t1 < gy1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
                "t1=%D not in ghosted [%D,%D) (ys=%D ym=%D gys=%D gym=%D)", t1,gy0,gy1, info.ys,info.ym, info.gys,info.gym);

      PetscScalar u_l[4];
	  
    u_l[0]=u[t0][xloc0][0];  
	  u_l[1]=u[t0][xloc1][0];
    u_l[2]=u[t1][xloc1][0];  
	  u_l[3]=u[t1][xloc0][0];

      /* -------- build 16×16 element matrix -------- */
      PetscScalar M[8][8] = {{0}};
      for (int i=0;i<4;++i){
        int r1 = 2*i; int r2 = r1+1;
        for (int j=0;j<4;++j){
          int cu = 2*j; int cv = cu+1;
          /* linear part  */
          
          /* eqn 1 */
          M[r1][cu] += A_space[i][j];
          M[r1][cv] += A_time [i][j];

          /* eqn 2 */
          M[r2][cu] += -A_time [i][j];
          M[r2][cv] += A_standard[i][j];

        }
      }

      /* -------- assemble – local indices -------- */

    MatSetValuesStencil(J,8,row,8,col,&M[0][0],ADD_VALUES);
    if (P!=J) MatSetValuesStencil(P,8,row,8,col,&M[0][0],ADD_VALUES);
    
	
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
     and do nothing at the boundaries for this wave eqn problem.
   *------------------------------------------------------------------ */
 
  /* ------------------------------------------------------------------ *
    Impose t=0 initial-condition rows for both u and v
  *------------------------------------------------------------------ */

  PetscInt    nlocal = 0;
  MatStencil *bc     = NULL;

  /* how many BC rows on this rank? */
  PetscInt count = 0;
  if (info.ys == 0) {
    count += 2 * info.xm;  /* 2 dofs (u,v) per x-node */
  }

  if (count > 0) {
    PetscMalloc1(count, &bc);
    PetscInt idx = 0;

    if (info.ys == 0) {
      for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) 
      {
        /* u at t=0 */
        bc[idx].i = x;
        bc[idx].j = 0;
        bc[idx].k = 0; 
        bc[idx].c = 0;
        ++idx;
        /* v at t=0 */
        bc[idx].i = x;
        bc[idx].j = 0;
        bc[idx].k = 0; 
        bc[idx].c = 1;
        ++idx;
      }
    }

    nlocal = idx;
  }

  /* Zero those rows and put 1 on the diagonal */
  MatZeroRowsStencil(J, nlocal, bc, 1.0, NULL, NULL);
  if (P != J) MatZeroRowsStencil(P, nlocal, bc, 1.0, NULL, NULL);

  if (bc) PetscFree(bc);

  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  if (P != J) {
    MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);
  }


    PetscFunctionReturn(0);
}