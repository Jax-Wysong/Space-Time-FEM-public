#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "appctx.h"
#include "ic.h"
#include "stiffness.h"
#include "residual.h"
#include "jacobian.h"
#include "asm.h"
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
  user.nx   = 100;  user.nt   = 100;
  user.xL   = -5.0;  user.xR   = 5.0;
  user.t0   = 0.0;  user.tF   = 10.0;
  user.IC = 0;		
  user.Nsub = 1;
  user.overlap = 1;
  user.interface_width = 1; 
  user.slab_pc_ras = PETSC_FALSE;
  user.full_overlap_dirichlet = PETSC_FALSE;
  user.first_sub_only_bc = PETSC_FALSE;

  user.dump_matrices = PETSC_FALSE;



  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,
                    "Space-time solver options",NULL);
  PetscOptionsInt  ("-nx"   ,"# x-nodes"      ,"" ,user.nx   ,&user.nx   ,NULL);
  PetscOptionsInt  ("-nt"   ,"# t-nodes"      ,"" ,user.nt   ,&user.nt   ,NULL);
  PetscOptionsInt  ("-IC"   ,"# initial conditions","" ,user.IC   ,&user.IC,NULL);
  PetscOptionsInt("-Nsub","number of time slabs","",user.Nsub,&user.Nsub,NULL);
  PetscOptionsInt("-overlap","overlap in time levels","",user.overlap,&user.overlap,NULL);
  PetscOptionsInt("-interface_width","width of interface","",user.interface_width,&user.interface_width,NULL);
  PetscOptionsBool("-slab_pc_ras","use RAS slab PC","",user.slab_pc_ras,&user.slab_pc_ras,NULL);
  PetscOptionsBool("-full_overlap_dirichlet","pin all overlap time levels as left Dirichlet (default: pin only 1)","",user.full_overlap_dirichlet,&user.full_overlap_dirichlet,NULL);
  PetscOptionsBool("-first_sub_only_bc","impose temporal Dirichlet BC only on the first subdomain, none on others","",user.first_sub_only_bc,&user.first_sub_only_bc,NULL);
  PetscOptionsReal ("-xL"   ,"left  x"        ,"" ,user.xL   ,&user.xL   ,NULL);
  PetscOptionsReal ("-xR"   ,"right x"        ,"" ,user.xR   ,&user.xR   ,NULL);
  PetscOptionsReal ("-t0"   ,"initial t"      ,"" ,user.t0   ,&user.t0   ,NULL);
  PetscOptionsReal ("-tF"   ,"final   t"      ,"" ,user.tF   ,&user.tF   ,NULL);
  PetscOptionsBool("-dump_matrices","dump J and M^{-1}J to binary files","",user.dump_matrices,&user.dump_matrices,NULL);
  PetscOptionsEnd();


  PetscInt swidth = user.overlap + user.interface_width;
  if (swidth == 0){ swidth = 1;}
  PetscPrintf(PETSC_COMM_WORLD, "Using overlap %d and interface width %d, so stencil width = %d\n", user.overlap, user.interface_width, swidth);
  /* ------------ build the DM ------------ */
  DM  dm;
  DMDACreate2d(PETSC_COMM_WORLD,                /* communicator          */
               DM_BOUNDARY_NONE,                /* x boundary */
               DM_BOUNDARY_NONE,                /* t boundary            */
               DMDA_STENCIL_BOX,                /* stencil               */
               user.nx, user.nt,                /* global grid           */
               PETSC_DECIDE,PETSC_DECIDE,       /* owner splits          */
               /* ----- dof  ---- */ 2,         /* (phi,u,chi,v) per node      */
               /* stencil width */ swidth,   /* need +1 in x,+1 in t  */
               NULL,NULL,                       /* no custom d-grid      */
               &dm);
			   
  DMSetFromOptions(dm);   /* allow -da_* flags   */
  DMDASetStencilType(dm,DMDA_STENCIL_BOX);  /* override STAR if still there */
  DMSetUp(dm);
  ierr = DMDASetFieldName(dm,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(dm,1,"v");CHKERRQ(ierr);
  
	if(user.IC == 0)
	{
		PetscPrintf(PETSC_COMM_WORLD, "\n!!! FORGOT TO SPECIFY INITIAL CONDITION !!!\n");
	}


  user.hx = (user.xR-user.xL)/(user.nx-1);
  user.ht = (user.tF-user.t0)/(user.nt-1);
  user.L  = user.xR - user.xL;
  Compute_linear_stiffness(user.A_time, user.A_space, user.A_standard, user.hx, user.ht);
  user.dm = dm;           /* <-- FormResidual/Jacobian read it */


  /* ------------ create SNES + work vectors/mats ------------ */
  
  Mat J;
  DMCreateMatrix(dm, &J);
  MatSetBlockSize(J, 2);
  
  SNES snes;
  SNESCreate(PETSC_COMM_WORLD,&snes);
  SNESSetDM(snes,dm);                       /* tells SNES to use dm */
  SNESSetFunction(snes,NULL,FormResidual,&user);
  SNESSetJacobian(snes,J,J,FormJacobian,&user);
  SNESSetFromOptions(snes);                 /* allow command line options to change snes options */


  /* ------------ set up preconditioner  ------------ */
  KSP ksp;
  PC  pc;
  PetscBool use_shell = PETSC_FALSE;
  PetscOptionsGetBool(NULL,NULL,"-use_shell_pc",&use_shell,NULL);


  SNESGetKSP(snes, &ksp);
  KSPSetDM(ksp, dm);
  KSPSetDMActive(ksp, PETSC_FALSE);  
  KSPGetPC(ksp, &pc);

  if (use_shell) {
    PCSetType(pc, PCSHELL);

    SampleShellPC *shell;
    PetscNew(&shell);
    shell->user = &user;
    shell->dm   = user.dm;
    shell->snes = snes;
    shell->use_ras = user.slab_pc_ras;

    PCShellSetContext(pc, shell);
    PCShellSetSetUp(pc, PCSetUp_SampleShell);
    PCShellSetApply(pc, PCApply_SampleShell);
    PCShellSetDestroy(pc, PCDestroy_SampleShell);
  }
  /* else: do nothing; -pc_type lu etc. will work */


  /* ------------ create vectors ------------ */
  Vec U, F, b;
  ierr = DMCreateGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&F);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&b);CHKERRQ(ierr);

  /* ------------ fill initial conditions on rank-local part ------------ */
  ierr = VecZeroEntries(U);CHKERRQ(ierr);
  // {
  //   DMDALocalInfo info;  DMDAGetLocalInfo(dm,&info);
  //   PetscInt xs=info.xs, xm=info.xm;
  //   PetscInt ys=info.ys, ym=info.ym;
  //   PetscScalar (**u)[2];
  //   DMDAVecGetArray(dm,U,&u);
  //   for (PetscInt t=ys; t<ys+ym; ++t){
  //     for (PetscInt x=xs; x<xs+xm; ++x){
  //       PetscReal x_phys = user.xL + x*user.hx;
  //       if (t==0){                                 /* IC at t=0      */
  //         if(user.IC == 1)
  //         {
  //           u[t][x][0] = wave_IC_u(x_phys); /* initial conditions found from solution */
  //           u[t][x][1] = wave_IC_v(x_phys);
  //         }
  //       }else{
  //         u[t][x][0] = 0.0;
  //         u[t][x][1] = 0.0;
  //       }
  //     }
  //   }
  //   DMDAVecRestoreArray(dm,U,&u);
  // }

  /* ------------ solve ------------ */
  
	PetscPrintf(PETSC_COMM_WORLD, "\n=================== Linear Wave Problem ===================\n\nUsing nx = %d, nt = %d\n", (int)user.nx, (int)user.nt);
	PetscPrintf(PETSC_COMM_WORLD, "xL = %g, xR = %g\nt0 = %g, tF = %g\n", user.xL, user.xR, user.t0, user.tF);
	PetscPrintf(PETSC_COMM_WORLD, "hx = %.8e, ht = %.8e\n\n", (double)user.hx, (double)user.ht);

    PetscPrintf(PETSC_COMM_WORLD,
      "=== Solve (T=[%.3f,%.3f], nt=%d) ===\n", (double)user.t0,(double)user.tF,user.nt);

  /* ------------ assemble J and build RHS b ------------ */
  /* Assemble J at the current U */
  ierr = FormJacobian(snes, U, J, J, &user);CHKERRQ(ierr);

  /* Compute F(U). F(U) = J U - b. */
  ierr = FormResidual(snes, U, F, &user);CHKERRQ(ierr);

  /* b = J*U - F */
  ierr = MatMult(J, U, b);CHKERRQ(ierr);
  ierr = VecAXPY(b, -1.0, F);CHKERRQ(ierr);

  /* ------------ solve J * U = b with KSP directly ------------ */
  ierr = KSPSetOperators(ksp, J, J);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  /* ------------ optional matrix dump for eigenvalue analysis ------------ */
  if (user.dump_matrices) {
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscInt N;
    MatGetSize(J, &N, NULL);

    /* 1. Dump J in PETSc binary format (readable by petsc4py) */
    {
      char fname_J[64];
      snprintf(fname_J, sizeof(fname_J), "J_%d-Nsub_%dx%d.dat", (int)user.Nsub, (int)user.nx, (int)user.nt);
      PetscViewer vJ;
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname_J, FILE_MODE_WRITE, &vJ);
      MatView(J, vJ);
      PetscViewerDestroy(&vJ);
      PetscPrintf(PETSC_COMM_WORLD, "Dumped J (%d x %d) to %s\n", (int)N, (int)N, fname_J);
    }

    /* 2. Build M^{-1}J column-by-column; write dense binary (int32 N, float64 N*N) on rank 0 */
    {
      PC pc_local;
      KSPGetPC(ksp, &pc_local);

      Vec e_i, Je_i, MiJe_i, seq_vec;
      VecScatter scatter;
      VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &e_i);
      VecDuplicate(e_i, &Je_i);
      VecDuplicate(e_i, &MiJe_i);
      VecScatterCreateToZero(MiJe_i, &scatter, &seq_vec);

      PetscScalar *dense_PA = NULL;
      if (rank == 0) PetscMalloc1((size_t)N * N, &dense_PA);

      for (PetscInt col = 0; col < N; col++) {
        VecZeroEntries(e_i);
        VecSetValue(e_i, col, 1.0, INSERT_VALUES);
        VecAssemblyBegin(e_i); VecAssemblyEnd(e_i);
        MatMult(J, e_i, Je_i);
        PCApply(pc_local, Je_i, MiJe_i);
        VecScatterBegin(scatter, MiJe_i, seq_vec, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd  (scatter, MiJe_i, seq_vec, INSERT_VALUES, SCATTER_FORWARD);
        if (rank == 0) {
          const PetscScalar *arr;
          VecGetArrayRead(seq_vec, &arr);
          for (PetscInt row = 0; row < N; row++)
            dense_PA[col * N + row] = arr[row];  /* column-major */
          VecRestoreArrayRead(seq_vec, &arr);
        }
      }

      char fname_PA[64];
      snprintf(fname_PA, sizeof(fname_PA), "PA_dense_%d-Nsub_%dx%d.bin", (int)user.Nsub, (int)user.nx, (int)user.nt);
      if (rank == 0) {
        FILE *fp = fopen(fname_PA, "wb");
        fwrite(&N,       sizeof(PetscInt),    1,             fp);
        fwrite(dense_PA, sizeof(PetscScalar), (size_t)N * N, fp);
        fclose(fp);
        PetscFree(dense_PA);
      }
      PetscPrintf(PETSC_COMM_WORLD,
                  "Dumped M^{-1}J (%d x %d) to %s\n", (int)N, (int)N, fname_PA);

      VecScatterDestroy(&scatter);
      VecDestroy(&seq_vec);
      VecDestroy(&e_i); VecDestroy(&Je_i); VecDestroy(&MiJe_i);
    }
  }

  PetscPrintf(PETSC_COMM_WORLD, "\n=== KSP solve (no SNES updates) ===\n");
  ierr = KSPSolve(ksp, b, U);CHKERRQ(ierr);
	
  /* calculate absolute residual norm*/
  Vec r;
  VecDuplicate(b, &r);
  MatMult(J, U, r);
  VecAXPY(r, -1.0, b);
  PetscReal norm_r;
  VecNorm(r, NORM_2, &norm_r);
  PetscPrintf(PETSC_COMM_WORLD, "Absolute residual norm = %.8e\n", (double)norm_r);

  /* Parallel-safe L2 and max error norms for u */
  {
    DMDALocalInfo info; DMDAGetLocalInfo(dm,&info);
    /* Local accumulators */
    PetscReal l2_u_local = 0.0;
    PetscReal max_u_local = 0.0;

    /* Access distributed array with dof */
    PetscScalar (**a)[2];  /* a[t][x][comp] */
    DMDAVecGetArrayRead(dm, U, &a);

    /* Global starts for physical coords */
    for (PetscInt tj = info.ys; tj < info.ys + info.ym; ++tj) {
      PetscReal t_phys = user.t0 + tj * user.ht;
      for (PetscInt xi = info.xs; xi < info.xs + info.xm; ++xi) {
        PetscReal x_phys = user.xL + xi * user.hx;

        /* analytic solutions */
        PetscReal u_exact = PetscExpReal(-(x_phys - PetscCosReal(t_phys)) * (x_phys - PetscCosReal(t_phys)));

        /* Extract solution values (dof 0 = u) */
        PetscReal diff_u = PetscRealPart(a[tj][xi][0]) - u_exact;

        l2_u_local += diff_u * diff_u;

        PetscReal abs_u = PetscAbsReal(diff_u);

        if (abs_u > max_u_local) max_u_local = abs_u;
      }
    }

    DMDAVecRestoreArrayRead(dm, U, &a);

    /* Global reductions */
    PetscReal l2_u_global = 0.0;
    PetscReal max_u_global = 0.0;

    MPIU_Allreduce(&l2_u_local, &l2_u_global, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
    MPIU_Allreduce(&max_u_local, &max_u_global, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD);

    /* Scale for discrete L2 norm over the (t,x) grid */
    PetscReal error_L2_u = PetscSqrtReal(l2_u_global * user.hx * user.ht);

    PetscPrintf(PETSC_COMM_WORLD, "U  L2 error norm  = %.8e\n", (double)error_L2_u);
    PetscPrintf(PETSC_COMM_WORLD, "U  max error norm = %.8e\n", (double)max_u_global);
  }




  /* ---------------------------------------------------------------------
              Dump field snapshots to a MATLAB script
    --------------------------------------------------------------------- */
  PetscInt save = 0;

  if (save) {
    /* U -> natural ordering (still distributed). x-fastest, then t */
    Vec Unat;
    DMDACreateNaturalVector(user.dm, &Unat);
    DMDAGlobalToNaturalBegin(user.dm, U, INSERT_VALUES, Unat);
    DMDAGlobalToNaturalEnd  (user.dm, U, INSERT_VALUES, Unat);
    VecSetBlockSize(Unat, 2); /* enforce bs=2: [u,v] */

    /* Manually extract components 0 (u) and 1 (v) into distributed Vecs */
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
      PetscInt g = bstart + i;                /* global index in u/v (0..nx*nt-1) */
      PetscScalar vu = ua[2*i + 0];
      PetscScalar vv = ua[2*i + 1];
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
            "MMS_IC_%d_1x%.5g_%dx%d.dat",(int)user.IC, (double)user.tF, (int)user.nx, (int)user.nt);		
    }


    PetscViewer viewer = NULL;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname , FILE_MODE_WRITE, &viewer);
    /* Write what you need */
    VecView(u,     viewer);
    VecView(vx,      viewer);
    VecView(vt,      viewer);

    PetscViewerDestroy(&viewer);

    /* 6) Cleanup */
    VecDestroy(&vx); VecDestroy(&vt);
    VecDestroy(&u);
    VecDestroy(&Unat);

    PetscPrintf(PETSC_COMM_WORLD,"Saved (1+1) u, x, t to PETSc binary.\n");

  }





    /* ------------ clean up ------------ */
  SNESDestroy(&snes); VecDestroy(&U); MatDestroy(&J); DMDestroy(&dm);
  PetscFinalize();
  return 0;
}
