#include "dm_create.h"
#include "stiffness.h"
#include "jacobian.h"
#include "residual.h"
#include "appctx.h"

PetscErrorCode CreateDMSNESForGrid(AppCtx *user, PetscInt nx, PetscInt nt, DM *dm, SNES *snes, Mat *J, Vec *U)
{
  PetscFunctionBeginUser;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                         DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX,
                         nx, nt, PETSC_DECIDE, PETSC_DECIDE,
                         4, 1, NULL, NULL, dm));
  PetscCall(DMDASetStencilType(*dm, DMDA_STENCIL_BOX));
  PetscCall(DMSetUp(*dm));

  user->nx = nx;
  user->nt = nt;
  user->hx = (user->xR - user->xL)/nx;
  user->ht = (user->tF - user->t0)/((PetscReal)nt - 1.0);
  user->L  = user->xR - user->xL;
  user->dm = *dm;
  Compute_linear_stiffness(user->A_time, user->A_space, user->A_standard,
                           user->hx, user->ht);

  PetscCall(DMCreateMatrix(*dm, J));
  PetscCall(DMCreateGlobalVector(*dm, U));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, snes));
  PetscCall(SNESSetDM(*snes, *dm));
  PetscCall(SNESSetFunction(*snes, NULL, FormResidual, user));
  PetscCall(SNESSetJacobian(*snes, *J, *J, FormJacobian, user));
  PetscCall(SNESSetFromOptions(*snes));

  PetscFunctionReturn(0);
}
