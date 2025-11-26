#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>

/*
These are the 2+1 extensions of the initial conditions for the
1. Plane Waves
2. Gaussian Packets
4. Phase-correlated plane waves
5. Oscillons
*/

/* ======================= Wave -IC 1 Plane Waves ======================= */
static inline PetscScalar phi_wave_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal C){	
	PetscReal L    	= 1.0;
	PetscReal xphi 	= 0.0;
    PetscReal yphi  = 0.0;
	PetscReal Aphi 	= A;
	PetscReal kphi 	= C*2.0*PETSC_PI/L;
	PetscReal kxphi	= kphi;
    PetscReal kyphi	= kphi;
  return Aphi * PetscSinReal(kxphi*(x-xphi) + kyphi*(y-yphi));
} 

static inline PetscScalar u_wave_IC(PetscReal x, PetscReal y, PetscReal mphi2, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xphi = 0.0;
	PetscReal yphi = 0.0;
	PetscReal Aphi = A;
	PetscReal kphi = C*2.0*PETSC_PI/L;
	PetscReal kxphi	= kphi;
	PetscReal kyphi = kphi;
	PetscReal wphi = -PetscSqrtReal(kxphi*kxphi + kyphi*kyphi + mphi2);
  return -wphi * Aphi * PetscCosReal(kxphi*(x-xphi) + kyphi*(y-yphi));           
}

static inline PetscScalar chi_wave_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xchi = L/3.0;
	PetscReal ychi = L/3.0;
	PetscReal Achi = A;
	PetscReal kchi = C*2.0*(2.0*PETSC_PI/L);
	PetscReal kxchi	= kchi;
	PetscReal kychi = kchi;
  return Achi * PetscSinReal(kxchi*(x-xchi) + kychi*(y-ychi));
}

static inline PetscScalar v_wave_IC(PetscReal x, PetscReal y, PetscReal mchi2, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xchi = L/3.0;
	PetscReal ychi = L/3.0;
	PetscReal Achi = A;
	PetscReal kchi = C*2.0*(2.0*PETSC_PI/L);
	PetscReal kxchi	= kchi;
	PetscReal kychi = kchi;
	PetscReal wchi = PetscSqrtReal(kxchi*kxchi + kychi*kychi + mchi2);
  return -wchi * Achi * PetscCosReal(kxchi*(x-xchi) + kychi*(y-ychi));
}


/* ======================= Guassian packet -IC 2 ======================= */

static inline PetscScalar phi_gauss_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xphi  = 0.3*Lx;
  PetscReal yphi  = 0.5*Ly;
  PetscReal Aphi  = A;
  PetscReal Cx 	  = C;
  PetscReal Cy    = C;
  PetscReal kxphi = Cx*2.0*PETSC_PI/Lx;
  PetscReal kyphi = Cy*2.0*PETSC_PI/Ly;
  PetscReal kphi  = PetscSqrtReal(kxphi*kxphi + kyphi*kyphi);
  PetscReal lphi  = 1.0/(4.0*kphi);

  PetscReal dx = x - xphi;
  PetscReal dy = y - yphi;
  PetscReal r2 = dx*dx + dy*dy;

  return Aphi * PetscExpReal(-r2/(2.0*lphi*lphi));
}


static inline PetscScalar u_gauss_IC(PetscReal x, PetscReal y, PetscReal mphi2, PetscReal A, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xphi  = 0.3*Lx;
  PetscReal yphi  = 0.5*Ly;
  PetscReal Aphi  = A;
  PetscReal Cx 	  = C;
  PetscReal Cy    = C;
  PetscReal kxphi = Cx*2.0*PETSC_PI/Lx;
  PetscReal kyphi = Cy*2.0*PETSC_PI/Ly;
  PetscReal kphi  = PetscSqrtReal(kxphi*kxphi + kyphi*kyphi);
  PetscReal lphi  = 1.0/(4.0*kphi);
  PetscReal cphi  = 1.0;

  PetscReal dx = x - xphi;
  PetscReal dy = y - yphi;
  PetscReal r2 = dx*dx + dy*dy;

  PetscReal wphi = Aphi * cphi * (dx + dy) / (lphi*lphi);
  return wphi * PetscExpReal(-r2/(2.0*lphi*lphi));
}

static inline PetscScalar chi_gauss_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xchi  = 0.7*Lx;
  PetscReal ychi  = 0.5*Ly;
  PetscReal Achi  = A;
  PetscReal Cx 	  = C;
  PetscReal Cy    = C;
  PetscReal kxchi = Cx*2.0*(2.0*PETSC_PI/Lx);
  PetscReal kychi = Cy*2.0*(2.0*PETSC_PI/Ly);
  PetscReal kchi  = PetscSqrtReal(kxchi*kxchi + kychi*kychi);
  PetscReal lchi  = 1.0/(4.0*kchi);

  PetscReal dx = x - xchi;
  PetscReal dy = y - ychi;
  PetscReal r2 = dx*dx + dy*dy;

  return Achi * PetscExpReal(-r2/(2.0*lchi*lchi));
}

static inline PetscScalar v_gauss_IC(PetscReal x, PetscReal y, PetscReal mchi2, PetscReal A, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xchi  = 0.7*Lx;
  PetscReal ychi  = 0.5*Ly;
  PetscReal Achi  = A;
  PetscReal Cx 	  = C;
  PetscReal Cy    = C;
  PetscReal kxchi = Cx*2.0*(2.0*PETSC_PI/Lx);
  PetscReal kychi = Cy*2.0*(2.0*PETSC_PI/Ly);
  PetscReal kchi  = PetscSqrtReal(kxchi*kxchi + kychi*kychi);
  PetscReal lchi  = 1.0/(4.0*kchi);
  PetscReal cchi  = -1.0; // Packets move opposite eachother (-1) Packets move with eachother (+1)

  PetscReal dx = x - xchi;
  PetscReal dy = y - ychi;
  PetscReal r2 = dx*dx + dy*dy;

  PetscReal wchi = Achi * cchi * (dx + dy) / (lchi*lchi);
  return wchi * PetscExpReal(-r2/(2.0*lchi*lchi));
}


/* ======================= Phase-correlated two-field plane waves -IC 4 ========================= */
static inline PetscScalar phi_pw_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xphi  = 0.0;
  PetscReal yphi  = 0.0;

  PetscReal Cx    = C;
  PetscReal Cy    = C; // Hard-coded tie; split later if desired

  PetscReal kx    = Cx * 2.0 * PETSC_PI / Lx;
  PetscReal ky    = Cy * 2.0 * PETSC_PI / Ly;

  return A * PetscCosReal( kx*(x - xphi) + ky*(y - yphi) );
}

static inline PetscScalar u_pw_IC(PetscReal x, PetscReal y, PetscReal mphi2, PetscReal A, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xphi  = 0.0;
  PetscReal yphi  = 0.0;

  PetscReal Cx    = C;
  PetscReal Cy    = C; // Hard-coded tie; split later if desired

  PetscReal kx    = Cx * 2.0 * PETSC_PI / Lx;
  PetscReal ky    = Cy * 2.0 * PETSC_PI / Ly;
  PetscReal kabs2 = kx*kx + ky*ky;
  PetscReal wphi  = PetscSqrtReal(kabs2 + mphi2);

  return A * wphi * PetscSinReal( kx*(x - xphi) + ky*(y - yphi) );
}

static inline PetscScalar chi_pw_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal r, PetscReal dphi, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xchi  = Lx/3.0;
  PetscReal ychi  = Ly/3.0;

  PetscReal Cx    = C;
  PetscReal Cy    = C; // Hard-coded tie; split later if desired

  PetscReal kx    = Cx * 2.0 * PETSC_PI / Lx;
  PetscReal ky    = Cy * 2.0 * PETSC_PI / Ly;

  return A * r * PetscCosReal( kx*(x - xchi) + ky*(y - ychi) + dphi );
}

static inline PetscScalar v_pw_IC(PetscReal x, PetscReal y, PetscReal mchi2, PetscReal A, PetscReal r, PetscReal dphi, PetscInt sigma, PetscReal C)
{
  PetscReal Lx    = 1.0;
  PetscReal Ly    = 1.0;
  PetscReal xchi  = Lx/3.0;
  PetscReal ychi  = Ly/3.0;

  PetscReal Cx    = C;
  PetscReal Cy    = C; // Hard-coded tie; split later if desired

  PetscReal kx    = Cx * 2.0 * PETSC_PI / Lx;
  PetscReal ky    = Cy * 2.0 * PETSC_PI / Ly;
  PetscReal kabs2 = kx*kx + ky*ky;
  PetscReal wchi  = PetscSqrtReal(kabs2 + mchi2);

  return (PetscScalar)( (PetscReal)sigma * A * r * wchi * PetscSinReal( kx*(x - xchi) + ky*(y - ychi) + dphi ) );
}


/* ======================= Oscillon-like, time-symmetric -IC 5 (2+1) ======================= */
static inline PetscReal sech(PetscReal z)
{
    return 1.0 / PetscCoshReal(z);
}


/* Optional 2D carrier factor: cos(kx*(x-x0) + ky*(y-y0)) if any component nonzero, else 1 */
static inline PetscReal carrier2D(PetscReal x, PetscReal y, PetscReal x0, PetscReal y0, PetscReal C)
{
    PetscReal Lx = 1.0, Ly = 1.0; /* if you change domain lengths, update here */
    PetscReal Cx0 = C;
    PetscReal Cy0 = C;
    PetscReal kx0 = Cx0 * 2.0 * PETSC_PI / Lx;
    PetscReal ky0 = Cy0 * 2.0 * PETSC_PI / Ly;
    if (kx0 != 0.0 || ky0 != 0.0) {
        return PetscCosReal( kx0*(x - x0) + ky0*(y - y0) );
    } else {
        return 1.0;
    }
}


static inline PetscScalar phi_osc_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal x0, PetscReal y0, PetscReal sigma, PetscReal C)
{
    PetscReal dx = x - x0;
    PetscReal dy = y - y0;
    PetscReal Cx0 = C;
    PetscReal Cy0 = C;
    PetscReal r = PetscSqrtReal(dx*dx + dy*dy);
    return (PetscScalar)( A * sech( r / sigma ) * carrier2D(x,y,x0,y0,C) );
}


static inline PetscScalar u_osc_IC(PetscReal x, PetscReal y, PetscReal mphi2, PetscReal A, PetscReal x0, PetscReal y0, PetscReal sigma, PetscReal C)
{
    PetscReal Cx0 = C;
    PetscReal Cy0 = C;
    (void)x; (void)y; (void)mphi2; (void)A; (void)x0; (void)y0; (void)sigma; (void)Cx0; (void)Cy0;
    return 0.0; /* time-symmetric */
}


static inline PetscScalar chi_osc_IC(PetscReal x, PetscReal y, PetscReal A, PetscReal r, PetscReal x0, PetscReal y0, PetscReal sigma, PetscReal dphi, PetscReal C)
{
    PetscReal Cx0 = C;
    PetscReal Cy0 = C;
    PetscReal dx = x - x0;
    PetscReal dy = y - y0;
    PetscReal rxy= PetscSqrtReal(dx*dx + dy*dy);
    return (PetscScalar)( A * r * PetscCosReal(dphi) * sech( rxy / sigma ) * carrier2D(x,y,x0,y0,C) );
}


static inline PetscScalar v_osc_IC(PetscReal x, PetscReal y, PetscReal mchi2, PetscReal A, PetscReal r, PetscReal x0, PetscReal y0, PetscReal sigma, PetscReal dphi, PetscReal C)
{
    PetscReal Cx0 = C;
    PetscReal Cy0 = C;
    (void)x; (void)y; (void)mchi2; (void)A; (void)r; (void)x0; (void)y0; (void)sigma; (void)dphi; (void)Cx0; (void)Cy0;
    return 0.0; /* time-symmetric */
}



#endif /* IC_H */