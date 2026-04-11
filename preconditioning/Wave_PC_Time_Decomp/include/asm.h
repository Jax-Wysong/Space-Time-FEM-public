#ifndef ASM_H
#define ASM_H

#include "appctx.h"

PetscErrorCode PCSetUp_SampleShell(PC pc);
PetscErrorCode PCDestroy_SampleShell(PC pc);
PetscErrorCode PCApply_SampleShell(PC pc, Vec x, Vec y);
PetscErrorCode stiff2(Mat A, PetscInt nx, PetscInt nt, void *ctx, PetscBool impose_left_dirichlet, const PetscScalar *Uslab, PC pc);
// static PetscErrorCode FillUslabFromGlobalScatter(Vec Ucur, VecScatter scat, Vec Utmp, PetscInt nloc, PetscScalar **Uslab_out);
// static PetscErrorCode FillUslabFromGlobal(const Vec Ucur, const PetscInt *glindex, PetscInt nloc, PetscScalar **Uslab_out);
// static PetscErrorCode PackSlabFromDMDALocal(DM dm, Vec xloc, PetscInt nx, PetscInt t0_global, PetscInt t1_global, Vec x_slab);
// static PetscErrorCode AddBackInteriorToGlobal_DMDARows(DM dm, Vec y, PetscInt nx, PetscInt t0_global, PetscInt t_add0, PetscInt t_add1, Vec y_slab);    
// static PetscErrorCode PackSlabFromGlobal(DM dm, Vec g, Vec loc, PetscInt nx, PetscInt t0_global, PetscInt t1_global, Vec slab_out);
// static PetscErrorCode CreateLocalSlabObjects(SampleShellPC *shell, PetscInt nloc, const char *prefix, Mat *A, KSP *ksp, Vec *x, Vec *y);
// static PetscErrorCode UnpackSlabToDMDALocalAdd(DM dm, Vec yloc, PetscInt nx, PetscInt t0_global, PetscInt t1_global, Vec y_slab);
// static PetscErrorCode AddBackFullWindow_ASM(SampleShellPC *shell, Vec y_global, PetscInt nx, PetscInt t0_global, PetscInt t1_global, Vec y_slab);

#endif /* ASM_H */