#include "stiffness.h"


void Compute_linear_stiffness(PetscScalar A_time[4][4], PetscScalar A_space[4][4], PetscReal hx, PetscReal ht)
{
	  PetscScalar third = 1.0/3.0;
	  PetscScalar sixth = 1.0/6.0;
	  PetscScalar ninth = 1.0/9.0;
	  PetscScalar twvth = 1.0/12.0;
	  PetscScalar eteen = 1.0/18.0;
	  PetscScalar thrsx = 1.0/36.0;
	  PetscScalar ht_hx = ht/hx;
	  PetscScalar h2 = hx*ht; 

	////////// Space Derivative Term (phi * phi_x) ///////////
	  A_space[0][0] = -ht*sixth; A_space[0][1] = ht*sixth; A_space[0][2] = ht*twvth; A_space[0][3] = -ht*twvth;
	  A_space[1][0] = -ht*sixth; A_space[1][1] = ht*sixth; A_space[1][2] = ht*twvth; A_space[1][3] = -ht*twvth;
	  A_space[2][0] = -ht*twvth; A_space[2][1] = ht*twvth; A_space[2][2] = ht*sixth; A_space[2][3] = -ht*sixth;
	  A_space[3][0] = -ht*twvth; A_space[3][1] = ht*twvth; A_space[3][2] = ht*sixth; A_space[3][3] = -ht*sixth;
	//////// Time Derivative Term (u_t * phi) //////////////
	  A_time[0][0] = -hx*sixth; A_time[0][1] = -hx*twvth; A_time[0][2] = hx*twvth; A_time[0][3] = hx*sixth;
	  A_time[1][0] = -hx*twvth; A_time[1][1] = -hx*sixth; A_time[1][2] = hx*sixth; A_time[1][3] = hx*twvth;
	  A_time[2][0] = -hx*twvth; A_time[2][1] = -hx*sixth; A_time[2][2] = hx*sixth; A_time[2][3] = hx*twvth;
	  A_time[3][0] = -hx*sixth; A_time[3][1] = -hx*twvth; A_time[3][2] = hx*twvth; A_time[3][3] = hx*sixth;

	}
