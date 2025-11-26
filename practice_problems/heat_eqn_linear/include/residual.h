#ifndef RESIDUAL_H
#define RESIDUAL_H

#include "appctx.h"

PetscErrorCode FormResidual(SNES snes, Vec U, Vec R, void *ctx);

#endif /* RESIDUAL_H */