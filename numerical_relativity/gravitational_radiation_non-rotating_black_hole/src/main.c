#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "appctx.h"
#include "ic.h"
#include "stiffness.h"
#include "residual.h"
#include "jacobian.h"
#include "compute_rstar.h"
#include "potential.h"
#include <petscviewer.h>


/*----------------------------------------------------*
   Parallel driver with DMDA
 *----------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  /* ------------ user parameters + command-line opts ------------ */
  AppCtx user;
  user.nx   = 100;  user.nt   = 101;
  user.xL   = -5.0;  user.xR   = 5.0;
  user.t0   = 0.0;  user.tF   = 10.0;
  user.IC = 0;


  user.M = 1.0;
  user.ell = 1.0;		



  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
                    "Space-time solver options",NULL);
  PetscOptionsInt  ("-nx"   ,"# x-nodes"      ,"" ,user.nx   ,&user.nx   ,NULL);
  PetscOptionsInt  ("-nt"   ,"# t-nodes"      ,"" ,user.nt   ,&user.nt   ,NULL);
  PetscOptionsInt  ("-IC"   ,"# initial conditions","" ,user.IC   ,&user.IC,NULL);
  PetscOptionsReal ("-xL"   ,"left  x"        ,"" ,user.xL   ,&user.xL   ,NULL);
  PetscOptionsReal ("-xR"   ,"right x"        ,"" ,user.xR   ,&user.xR   ,NULL);
  PetscOptionsReal ("-t0"   ,"initial t"      ,"" ,user.t0   ,&user.t0   ,NULL);
  PetscOptionsReal ("-tF"   ,"final   t"      ,"" ,user.tF   ,&user.tF   ,NULL);
  PetscOptionsEnd();



  /* ------------ build the DM ------------ */
  DM  dm;
  DMDACreate2d(PETSC_COMM_WORLD,                /* communicator          */
               DM_BOUNDARY_NONE,                /* x boundary            */
               DM_BOUNDARY_NONE,                /* t boundary            */
               DMDA_STENCIL_BOX,                /* stencil               */
               user.nx, user.nt,                /* global grid           */
               PETSC_DECIDE,PETSC_DECIDE,       /* owner splits          */
               /* ----- dof  ---- */ 2,         /* (u,v) per node        */
               /* stencil width */ 1,           /* +1 in x,+1 in t       */
               NULL,NULL,                       /* no custom d-grid      */
               &dm);
			   
  DMSetFromOptions(dm);   /* allow -da_* flags   */
  DMDASetStencilType(dm,DMDA_STENCIL_BOX);  /* ← override STAR if still there */
  DMSetUp(dm);
  

	if(user.IC == 0)
	{
		PetscPrintf(PETSC_COMM_WORLD, "\n!!! FORGOT TO SPECIFY INITIAL CONDITION !!!\n");
	}


  user.hx = (user.xR-user.xL)/(user.nx-1);
  user.ht = (user.tF-user.t0)/(user.nt-1);
  user.L  = user.xR - user.xL;
  Compute_linear_stiffness(user.A_time, user.A_space, user.A_standard, user.A_time_edge_R, user.A_time_edge_L, user.hx, user.ht);
  user.dm = dm;           /* <-- FormResidual/Jacobian read it */

  /* get the r(x) and V(x) values */
  SetupBH(&user);

  /* figure out where the particle is in tortoise coord x */
  PetscReal x0 = 50.0;
  PetscReal v  = 0.5;
  user.x_p_of_t = malloc(user.nt * sizeof(PetscScalar));
  for (PetscInt n = 0; n < user.nt; ++n) {
    PetscReal t = n * user.ht;
    user.x_p_of_t[n] = x0 - v * t;   // store particle position in tortoise coord x
  }



  /* ------------ create SNES + work vectors/mats ------------ */
  
  Mat J;
  DMCreateMatrix(dm,&J);
  
  SNES snes;
  SNESCreate(PETSC_COMM_WORLD,&snes);
  SNESSetDM(snes,dm);                             /* tells SNES to use dm */
  SNESSetFunction(snes,NULL,FormResidual,&user);
  SNESSetJacobian(snes,J,J,FormJacobian,&user);  /* P == J */
  SNESSetFromOptions(snes);                      /* can change snes stuff from command line */

  Vec  U;   /* solution vector */
  DMCreateGlobalVector(dm,&U);     

  /* ------------ fill initial conditions on rank-local part ------------ */
  {
    DMDALocalInfo info;  DMDAGetLocalInfo(dm,&info);
    PetscInt xs=info.xs, xm=info.xm;
    PetscInt ys=info.ys, ym=info.ym;
    PetscScalar (**u)[2];
    DMDAVecGetArray(dm,U,&u);
    for (PetscInt t=ys; t<ys+ym; ++t){
      for (PetscInt x=xs; x<xs+xm; ++x){
        PetscReal x_phys = user.xL + x*user.hx;
        if (t==0){                                 /* IC at t=0      */
          if(user.IC == 1)
          {
            u[t][x][0] = BH_IC_u(x_phys); /*  both of these are 0.0 because we are */
            u[t][x][1] = BH_IC_v(x_phys); /*  interested in seeing what the source term does */
          }
        }else{
          u[t][x][0] = 0.0;
          u[t][x][1] = 0.0;
        }
      }
    }
    DMDAVecRestoreArray(dm,U,&u);
  }


  /* Print some information regarding the mesh and size of domain */
	PetscPrintf(PETSC_COMM_WORLD, "\n=================== Non-Rotating BH ===================\n\nUsing nx = %d, nt = %d\n", (int)user.nx, (int)user.nt);
	PetscPrintf(PETSC_COMM_WORLD, "xL = %g, xR = %g\nt0 = %g, tF = %g\n", user.xL, user.xR, user.t0, user.tF);
	PetscPrintf(PETSC_COMM_WORLD, "hx = %.8e, ht = %.8e\n\n", (double)user.hx, (double)user.ht);

    PetscPrintf(PETSC_COMM_WORLD,
      "=== Solve (T=[%.3f,%.3f], nt=%d) ===\n", (double)user.t0,(double)user.tF,user.nt);

      /* ------------ solve ------------ */
    SNESSolve(snes,NULL,U); /* solution vector is U */
	

  /* ---------------------------------------------------------------------
              Dump field snapshots to a MATLAB script
    --------------------------------------------------------------------- */
  PetscInt save = 1;

  if (save) {
    /* U -> natural ordering (still distributed). x-fastest, then t */
    Vec Unat;
    DMDACreateNaturalVector(user.dm, &Unat);
    DMDAGlobalToNaturalBegin(user.dm, U, INSERT_VALUES, Unat);
    DMDAGlobalToNaturalEnd  (user.dm, U, INSERT_VALUES, Unat);
    VecSetBlockSize(Unat, 2); /* enforce bs=2: [u, v] */

    /* Manually extract u into distributed Vec */
    PetscInt Nloc, Nglob; 
    VecGetLocalSize(Unat, &Nloc);
    VecGetSize(Unat, &Nglob);
    PetscInt nloc = Nloc/2, nglob = Nglob/2;  /* number of (x,t) points */

    Vec u;
    VecCreateMPI(PETSC_COMM_WORLD, nloc, nglob, &u);
    PetscObjectSetName((PetscObject)u, "u");

    const PetscScalar *ua;
    VecGetArrayRead(Unat, &ua);

    PetscInt urstart; VecGetOwnershipRange(Unat, &urstart, NULL);
    PetscInt bstart = urstart/2; /* global block index of local start */

    for (PetscInt i = 0; i < nloc; ++i) {
      PetscInt g = bstart + i;                /* global index in u (0..nx*nt-1) */
      PetscScalar vu = ua[2*i + 0];
      VecSetValues(u, 1, &g, &vu, INSERT_VALUES);
    }
    VecRestoreArrayRead(Unat, &ua);
    VecAssemblyBegin(u); VecAssemblyEnd(u);

    /* Coordinates x (nx) and time t (nt) as Vecs */
    Vec vx=NULL, vt=NULL;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nx, &vx);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, user.nt, &vt);
    PetscObjectSetName((PetscObject)vx, "x");
    PetscObjectSetName((PetscObject)vt, "t");

    PetscInt rs,re,i; PetscScalar val;
    VecGetOwnershipRange(vx,&rs,&re);
    for (i=rs;i<re;i++){ val = (PetscScalar)(user.xL + (i)*(user.hx)); VecSetValues(vx,1,&i,&val,INSERT_VALUES); }
    VecAssemblyBegin(vx); VecAssemblyEnd(vx);

    VecGetOwnershipRange(vt,&rs,&re);
    for (i=rs;i<re;i++){ val = (PetscScalar)(user.t0 + (i)*(user.ht)); VecSetValues(vt,1,&i,&val,INSERT_VALUES); }
    VecAssemblyBegin(vt); VecAssemblyEnd(vt);

    /* Viewer: PETSc binary */
      char fname[PETSC_MAX_PATH_LEN];
    
    if (user.IC == 1){
      PetscSNPrintf(fname,sizeof(fname),
            "BH_test_1x%.5g_%dx%d.dat",(double)user.tF, (int)user.nx, (int)user.nt);		
    }


    PetscViewer viewer = NULL;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname , FILE_MODE_WRITE, &viewer);
    /* view the x, t, and u vectors */
    VecView(vx, viewer);
    VecView(vt, viewer);
    VecView(u,     viewer);

    PetscViewerDestroy(&viewer);

    /* Cleanup */
    VecDestroy(&vx); VecDestroy(&vt);
    VecDestroy(&u); 
    VecDestroy(&Unat);

    PetscPrintf(PETSC_COMM_WORLD,"Saved x, t, and U to PETSc binary.\n");

  }





    /* ------------ clean up ------------ */
  SNESDestroy(&snes); VecDestroy(&U); MatDestroy(&J); DMDestroy(&dm);
  PetscFinalize();
  return 0;
}
