#ifndef IC_H
#define IC_H

#include <petscsys.h>
#include <petscmath.h>

/*
These are the initial conditions for the
1. Plane Waves
2. Gaussian Packets
3. Colored-noise
4. Phase-correlated plane waves
5. Oscillons
*/

/* ======================= Wave -IC 1 ======================= */
static inline PetscScalar phi_wave_IC(PetscReal x, PetscReal A, PetscReal C){	
	PetscReal L    	= 1.0;
	PetscReal xphi 	= 0.0;
	PetscReal Aphi 	= A;
	PetscReal kphi 	= C*2.0*PETSC_PI/L;
	PetscReal kxphi	= kphi;
  return Aphi * PetscSinReal(kxphi*(x-xphi));
} 

static inline PetscScalar u_wave_IC(PetscReal x, PetscReal mphi2, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xphi = 0.0;
	PetscReal Aphi = A;
	PetscReal kphi = C*2.0*PETSC_PI/L;
	PetscReal kxphi	= kphi;
	PetscReal wphi = -PetscSqrtReal(kxphi*kxphi + mphi2);
  return -wphi * Aphi * PetscCosReal(kxphi*(x-xphi));      
}

static inline PetscScalar chi_wave_IC(PetscReal x, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xchi = L/3.0;
	PetscReal Achi = A;
	PetscReal kchi = C*2.0*(2.0*PETSC_PI/L);
	PetscReal kxchi	= kchi;
  return Achi * PetscSinReal(kxchi*(x-xchi));
}

static inline PetscScalar v_wave_IC(PetscReal x, PetscReal mchi2, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xchi = L/3.0;
	PetscReal Achi = A;
	PetscReal kchi = C*2.0*(2.0*PETSC_PI/L);
	PetscReal kxchi	= kchi;
	PetscReal wchi = PetscSqrtReal(kxchi*kxchi + mchi2);
  return -wchi * Achi * PetscCosReal(kxchi*(x-xchi));
}


/* ======================= Guassian packet -IC 2 ======================= */

static inline PetscScalar phi_gauss_IC(PetscReal x, PetscReal A, PetscReal C){
	PetscReal L    	= 1.0;
	PetscReal xphi 	= 0.3*L;
	PetscReal Aphi 	= A;
	PetscReal kphi 	= C*2.0*PETSC_PI/L;
	PetscReal lphi 	= 1.0/(4.0*kphi);
  return Aphi * PetscExpReal(-(x-xphi)*(x-xphi)/(2*lphi*lphi));
} 

static inline PetscScalar u_gauss_IC(PetscReal x, PetscReal mphi2, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xphi = 0.3*L;
	PetscReal Aphi = A;
	PetscReal kphi = C*2.0*PETSC_PI/L;
	PetscReal lphi = 1.0/(4.0*kphi);
	PetscReal cphi = 1.0;
	PetscReal wphi = Aphi*cphi*(x-xphi)/(lphi*lphi);
  return wphi * PetscExpReal(-(x-xphi)*(x-xphi)/(2*lphi*lphi));      
}

static inline PetscScalar chi_gauss_IC(PetscReal x, PetscReal A, PetscReal C){
	PetscReal L    	= 1.0;
	PetscReal xchi 	= 0.7*L;
	PetscReal Achi 	= A;
	PetscReal kchi 	= C*2.0*(2.0*PETSC_PI/L);
	PetscReal lchi 	= 1.0/(4.0*kchi);
  return Achi * PetscExpReal(-(x-xchi)*(x-xchi)/(2*lchi*lchi));
}

static inline PetscScalar v_gauss_IC(PetscReal x, PetscReal mchi2, PetscReal A, PetscReal C){
	PetscReal L    = 1.0;
	PetscReal xchi = 0.7*L;
	PetscReal Achi = A;
	PetscReal kchi = C*2.0*(2.0*PETSC_PI/L);
	PetscReal lchi = 1.0/(4.0*kchi);
	PetscReal cchi = -1.0; // -1.0 -> counter-propagation ; 1.0 -> co-moving propagation
	PetscReal wchi = Achi*cchi*(x-xchi)/(lchi*lchi);
  return wchi * PetscExpReal(-(x-xchi)*(x-xchi)/(2*lchi*lchi));  
}

/* ======================= Colored-Noise Spectra -IC 3 ======================= */

/* Compute normalization C(A, ns) so RMS_x[phi] ~ A for random phases. */
static inline PetscReal Cnorm(PetscReal A, PetscReal ns, PetscInt n1, PetscInt n2, PetscReal L)
{
  const PetscReal twoPiOverL = 2.0 * PETSC_PI / L;
  PetscReal denom = 0.0;
  for (PetscInt n = n1; n <= n2; ++n) {
    const PetscReal kn = twoPiOverL * (PetscReal)n;
    denom += PetscPowReal(kn, ns);
  }
  denom *= 0.5;
  return A / PetscSqrtReal(denom);
}

/* omega(k) for each field (free-field dispersion) */
static inline PetscReal omega_k(PetscReal k, PetscReal m2)
{
  return PetscSqrtReal(k*k + m2);
}

static inline PetscScalar phi_noise_IC(PetscReal x, PetscReal A, PetscReal ns, PetscInt n1, PetscInt n2, PetscReal L, const PetscReal *theta) 
{
  const PetscReal twoPiOverL = 2.0 * PETSC_PI / L;
  const PetscReal Cphi = Cnorm(A, ns, n1, n2, L);

  PetscReal sum = 0.0;
  for (PetscInt n = n1; n <= n2; ++n) {
    const PetscReal kn = twoPiOverL * (PetscReal)n;
    const PetscReal phase = kn*x + theta[n - n1];
    sum += PetscPowReal(kn, 0.5*ns) * PetscCosReal(phase);
  }
  return (PetscScalar)(Cphi * sum);
}

static inline PetscScalar u_noise_IC(PetscReal x, PetscReal mphi2, PetscReal A, PetscReal ns, PetscInt n1, PetscInt n2, PetscReal L, const PetscReal *theta, const PetscInt  *s_sign)
{
  const PetscReal twoPiOverL = 2.0 * PETSC_PI / L;
  const PetscReal Cphi = Cnorm(A, ns, n1, n2, L);

  PetscReal sum = 0.0;
  for (PetscInt n = n1; n <= n2; ++n) {
    const PetscReal kn = twoPiOverL * (PetscReal)n;
    const PetscReal w  = omega_k(kn, mphi2);
    const PetscReal phase = kn*x + theta[n - n1];
    sum += (PetscReal)s_sign[n - n1] * w * PetscPowReal(kn, 0.5*ns) * PetscSinReal(phase);
  }
  /* Matches Eq. (4) once you take the x-derivative and cancel k_n. */
  return (PetscScalar)(Cphi * sum);
}

static inline PetscScalar chi_noise_IC(PetscReal x, PetscReal Achi, PetscReal ns_chi, PetscInt n1, PetscInt n2, PetscReal L, const PetscReal *theta, PetscReal dphi)
{
  const PetscReal twoPiOverL = 2.0 * PETSC_PI / L;
  const PetscReal Cchi = Cnorm(Achi, ns_chi, n1, n2, L);

  PetscReal sum = 0.0;
  for (PetscInt n = n1; n <= n2; ++n) {
    const PetscReal kn = twoPiOverL * (PetscReal)n;
    const PetscReal phase = kn*x + theta[n - n1] + dphi;
    sum += PetscPowReal(kn, 0.5*ns_chi) * PetscCosReal(phase);
  }
  return (PetscScalar)(Cchi * sum);
}

static inline PetscScalar v_noise_IC(PetscReal x, PetscReal mchi2, PetscReal Achi, PetscReal ns_chi, PetscInt n1, PetscInt n2, PetscReal L, const PetscReal *theta, const PetscInt  *s_sign, PetscReal dphi)
{
  const PetscReal twoPiOverL = 2.0 * PETSC_PI / L;
  const PetscReal Cchi = Cnorm(Achi, ns_chi, n1, n2, L);

  PetscReal sum = 0.0;
  for (PetscInt n = n1; n <= n2; ++n) {
    const PetscReal kn = twoPiOverL * (PetscReal)n;
    const PetscReal w  = omega_k(kn, mchi2);
    const PetscReal phase = kn*x + theta[n - n1] + dphi;
    sum += (PetscReal)s_sign[n - n1] * w * PetscPowReal(kn, 0.5*ns_chi) * PetscSinReal(phase);
  }
  return (PetscScalar)(Cchi * sum);
}

/* ======================= Phase-correlated two-field plane waves -IC 4 */
static inline PetscScalar phi_pw_IC(PetscReal x, PetscReal A)
{
	PetscReal L = 1.0;
	PetscReal k = 2.0 * PETSC_PI / L;
	PetscReal xphi 	= 0.0;
  return A * PetscCosReal(k*(x - xphi));
}

static inline PetscScalar u_pw_IC(PetscReal x, PetscReal mphi2, PetscReal A)
{
	PetscReal L = 1.0;
	PetscReal k = 2.0 * PETSC_PI / L;
	PetscReal xphi 	= 0.0;
  PetscReal wphi = PetscSqrtReal(k*k + mphi2);
  return A * wphi * PetscSinReal(k*(x - xphi));
}

static inline PetscScalar chi_pw_IC(PetscReal x, PetscReal A, PetscReal r, PetscReal dphi)
{
	PetscReal L = 1.0;
	PetscReal k = 2.0 * PETSC_PI / L;
	PetscReal xchi 	= L/3.0;
  return A * r * PetscCosReal(k*(x - xchi) + dphi);
}

static inline PetscScalar v_pw_IC(PetscReal x, PetscReal mchi2, PetscReal A, PetscReal r, PetscReal dphi, PetscInt sigma)
{
	PetscReal L = 1.0;
	PetscReal k = 2.0 * PETSC_PI / L;
	PetscReal xchi 	= L/3.0;
  PetscReal wchi = PetscSqrtReal(k*k + mchi2);
  return (PetscScalar)( (PetscReal)sigma * A * r * wchi * PetscSinReal(k*(x - xchi) + dphi) );
}


/* ======================= Oscillon-like, time-symmetric -IC 5 ======================= */
static inline PetscReal sech(PetscReal z) 
{
	return 1.0 / PetscCoshReal(z);
}

/* Optional carrier factor: cos(k0(x-x0)) if k0>0, else 1 */
static inline PetscReal carrier(PetscReal x, PetscReal x0, PetscReal k0)
{
  return (k0 > 0.0) ? PetscCosReal(2.0*PETSC_PI*(x - x0)) : 1.0; // SHOULD BE DIVIDED BY L BUT L IS 1.0!!
}

static inline PetscScalar phi_osc_IC(PetscReal x, PetscReal A, PetscReal x0, PetscReal sigma, PetscReal k0)
{
  return (PetscScalar)( A * sech( (x - x0)/sigma ) * carrier(x,x0,k0) );
}

static inline PetscScalar u_osc_IC(PetscReal x, PetscReal mphi2, PetscReal A, PetscReal x0, PetscReal sigma, PetscReal k0)
{
  (void)x; (void)mphi2; (void)A; (void)x0; (void)sigma; (void)k0;
  return 0.0; /* time-symmetric */
}

static inline PetscScalar chi_osc_IC(PetscReal x, PetscReal A, PetscReal r, PetscReal x0, PetscReal sigma, PetscReal dphi, PetscReal k0)
{
  return (PetscScalar)( A * r * PetscCosReal(dphi) * sech( (x - x0)/sigma ) * carrier(x,x0,k0) );
}

static inline PetscScalar v_osc_IC(PetscReal x, PetscReal mchi2, PetscReal A, PetscReal r, PetscReal x0, PetscReal sigma, PetscReal dphi, PetscReal k0)
{
  (void)x; (void)mchi2; (void)A; (void)r; (void)x0; (void)sigma; (void)dphi; (void)k0;
  return 0.0; /* time-symmetric */
}



#endif /* IC_H */