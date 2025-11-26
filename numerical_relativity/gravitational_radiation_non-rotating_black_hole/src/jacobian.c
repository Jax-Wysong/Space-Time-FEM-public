#include <stdlib.h>
#include "appctx.h"
#include "source.h"

/*==============================================================*
   Parallel Jacobian for a 2-D DMDA with 2 dofs/point
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
  MatStencil        row[8], col[8];
  PetscInt          xs,ys,xm,ym, nx,nt;
  PetscReal         hx,ht;
  PetscScalar     (**u)[2];                    /* ghosted view of U – 2 dofs/pt */

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
  PetscScalar (*A_standard)[4]= user->A_standard;
  PetscScalar (*A_time_edge_R)[4] = user->A_time_edge_R; 
  PetscScalar (*A_time_edge_L)[4] = user->A_time_edge_L;

  /* ------ Potential terms ------- */
  PetscScalar *Vx = user->V_of_x;


  /* --- Read-only access to a ghosted copy of U --- */
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
	  for (int comp = 0; comp < 2; ++comp, ++k) { /* only 2 dofs (u,v) */
		row[k].i = xglob[corner];
		row[k].j = tglob[corner];
		row[k].c = comp;
		col[k]   = row[k];
	  }
	} 


      /* --- Potential value for this element (average of endpoints) --- */
      PetscScalar V_left  = Vx[xElm];
      PetscScalar V_right = Vx[xElm+1];
      PetscScalar V_elem  = 0.5*(V_left + V_right);

      /* -------- build 8×8 element matrix -------- */
      PetscScalar M[8][8] = {{0}};
      for (int i=0;i<4;++i){
        int r1 = 2*i;
        int r2 = r1+1;
        for (int j=0;j<4;++j){
          int cu = 2*j;
          int cv = cu+1;
          /* linear part  */
          
          /* eqn 1: v_t - u_xx + Vu */
          M[r1][cu] += A_space[i][j] + V_elem * A_standard[i][j];
          M[r1][cv] += A_time [i][j];

          /* eqn 2: -u_t + v */
          M[r2][cu] += -A_time [i][j];
          M[r2][cv] += A_standard[i][j];

        }
      }

      /* --- Sommerfeld edge contributions (only on boundary elements) --- */

      if (xElm == 0) {
        /* left boundary: -int v w dt at x_L  */
        for (int i=0;i<4;++i){
          int r1 = 2*i;   // eqn1
          for (int j=0;j<4;++j){
            int cv = 2*j+1;   // v column
            M[r1][cv] -= A_time_edge_L[i][j];   
          }
        }
      }

      if (xElm == nx-2) {
        /* right boundary: -int v w dt at x_R */
        for (int i=0;i<4;++i){
          int r1 = 2*i;
          for (int j=0;j<4;++j){
            int cv = 2*j+1;
            M[r1][cv] -= A_time_edge_R[i][j];   
          }
        }
      }

      /* -------- nonlinear block -------- */
      /* not needed here for linear problem */

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
     Only the rank(s) that own t=0 have ys==0, so we do it locally then
     call MatZeroRowsLocal().
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
      for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {
        /* u at t=0 */
        bc[idx].i = x;
        bc[idx].j = 0;
        bc[idx].c = 0;
        ++idx;
        /* v at t=0 */
        bc[idx].i = x;
        bc[idx].j = 0;
        bc[idx].c = 1;
        ++idx;
      }
    }

    nlocal = idx;
  }

  /* Zero those rows & put 1 on the diagonal */
  MatZeroRowsStencil(J, nlocal, bc, 1.0, NULL, NULL);
  if (P != J) MatZeroRowsStencil(P, nlocal, bc, 1.0, NULL, NULL);

  PetscFree(bc);

  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  if (P != J) {
    MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);
  }


  PetscFunctionReturn(0);
}