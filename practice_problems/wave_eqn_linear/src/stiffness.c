#include "stiffness.h"


void Compute_linear_stiffness(PetscScalar A_time[4][4], PetscScalar A_space[4][4], PetscScalar A_standard[4][4], PetscReal hx, PetscReal ht)
{
	  PetscScalar third = 1.0/3.0;
	  PetscScalar sixth = 1.0/6.0;
	  PetscScalar ninth = 1.0/9.0;
	  PetscScalar twvth = 1.0/12.0;
	  PetscScalar eteen = 1.0/18.0;
	  PetscScalar thrsx = 1.0/36.0;
	  PetscScalar ht_hx = ht/hx; //Space term 
	  PetscScalar h = hx; // time term
	  PetscScalar h2 = hx*ht; //standard

	////////// Space Derivative Term (u_x * phi) ///////////
	  A_space[0][0] = ht_hx*third;  A_space[0][1] = -ht_hx*third; A_space[0][2] = -ht_hx*sixth; A_space[0][3] = ht_hx*sixth;
	  A_space[1][0] = -ht_hx*third; A_space[1][1] = ht_hx*third;  A_space[1][2] = ht_hx*sixth;  A_space[1][3] = -ht_hx*sixth;
	  A_space[2][0] = -ht_hx*sixth; A_space[2][1] = ht_hx*sixth;  A_space[2][2] = ht_hx*third;  A_space[2][3] = -ht_hx*third;
	  A_space[3][0] = ht_hx*sixth;  A_space[3][1] = -ht_hx*sixth; A_space[3][2] = -ht_hx*third; A_space[3][3] = ht_hx*third;
	////////// Time Derivative Term (u_t * phi) //////////////
	  A_time[0][0] = -h*sixth; A_time[0][1] = -h*twvth; A_time[0][2] = h*twvth; A_time[0][3] = h*sixth;
	  A_time[1][0] = -h*twvth; A_time[1][1] = -h*sixth; A_time[1][2] = h*sixth; A_time[1][3] = h*twvth;
	  A_time[2][0] = -h*twvth; A_time[2][1] = -h*sixth; A_time[2][2] = h*sixth; A_time[2][3] = h*twvth;
	  A_time[3][0] = -h*sixth; A_time[3][1] = -h*twvth; A_time[3][2] = h*twvth; A_time[3][3] = h*sixth;
	/////////// Mass Term (u * phi) ////////////
	  A_standard[0][0] = h2*ninth; A_standard[0][1] = h2*eteen; A_standard[0][2] = h2*thrsx; A_standard[0][3] = h2*eteen;
	  A_standard[1][0] = h2*eteen; A_standard[1][1] = h2*ninth; A_standard[1][2] = h2*eteen; A_standard[1][3] = h2*thrsx;
	  A_standard[2][0] = h2*thrsx; A_standard[2][1] = h2*eteen; A_standard[2][2] = h2*ninth; A_standard[2][3] = h2*eteen;
	  A_standard[3][0] = h2*eteen; A_standard[3][1] = h2*thrsx; A_standard[3][2] = h2*eteen; A_standard[3][3] = h2*ninth;
}