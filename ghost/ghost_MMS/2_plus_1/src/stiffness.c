#include "stiffness.h"

void Compute_linear_stiffness(PetscScalar A_time[8][8],
                              PetscScalar A_space_x[8][8],
                              PetscScalar A_space_y[8][8],
                              PetscScalar A_mass[8][8],
                              PetscReal hx, PetscReal hy, PetscReal ht)
{
    /* x-derivative term */
    A_space_x[0][0] =  ht*hy/( 9.0*hx);  A_space_x[0][1] = -ht*hy/( 9.0*hx);  A_space_x[0][2] = -ht*hy/(18.0*hx);  A_space_x[0][3] =  ht*hy/(18.0*hx);
    A_space_x[0][4] =  ht*hy/(18.0*hx);  A_space_x[0][5] = -ht*hy/(18.0*hx);  A_space_x[0][6] = -ht*hy/(36.0*hx);  A_space_x[0][7] =  ht*hy/(36.0*hx);

    A_space_x[1][0] = -ht*hy/( 9.0*hx);  A_space_x[1][1] =  ht*hy/( 9.0*hx);  A_space_x[1][2] =  ht*hy/(18.0*hx);  A_space_x[1][3] = -ht*hy/(18.0*hx);
    A_space_x[1][4] = -ht*hy/(18.0*hx);  A_space_x[1][5] =  ht*hy/(18.0*hx);  A_space_x[1][6] =  ht*hy/(36.0*hx);  A_space_x[1][7] = -ht*hy/(36.0*hx);

    A_space_x[2][0] = -ht*hy/(18.0*hx);  A_space_x[2][1] =  ht*hy/(18.0*hx);  A_space_x[2][2] =  ht*hy/( 9.0*hx);  A_space_x[2][3] = -ht*hy/( 9.0*hx);
    A_space_x[2][4] = -ht*hy/(36.0*hx);  A_space_x[2][5] =  ht*hy/(36.0*hx);  A_space_x[2][6] =  ht*hy/(18.0*hx);  A_space_x[2][7] = -ht*hy/(18.0*hx);

    A_space_x[3][0] =  ht*hy/(18.0*hx);  A_space_x[3][1] = -ht*hy/(18.0*hx);  A_space_x[3][2] = -ht*hy/( 9.0*hx);  A_space_x[3][3] =  ht*hy/( 9.0*hx);
    A_space_x[3][4] =  ht*hy/(36.0*hx);  A_space_x[3][5] = -ht*hy/(36.0*hx);  A_space_x[3][6] = -ht*hy/(18.0*hx);  A_space_x[3][7] =  ht*hy/(18.0*hx);

    A_space_x[4][0] =  ht*hy/(18.0*hx);  A_space_x[4][1] = -ht*hy/(18.0*hx);  A_space_x[4][2] = -ht*hy/(36.0*hx);  A_space_x[4][3] =  ht*hy/(36.0*hx);
    A_space_x[4][4] =  ht*hy/( 9.0*hx);  A_space_x[4][5] = -ht*hy/( 9.0*hx);  A_space_x[4][6] = -ht*hy/(18.0*hx);  A_space_x[4][7] =  ht*hy/(18.0*hx);

    A_space_x[5][0] =  -ht*hy/(18.0*hx);  A_space_x[5][1] = ht*hy/(18.0*hx);  A_space_x[5][2] = ht*hy/(36.0*hx);  A_space_x[5][3] =  -ht*hy/(36.0*hx);
    A_space_x[5][4] =  -ht*hy/(9.0*hx);  A_space_x[5][5] = ht*hy/(9.0*hx);  A_space_x[5][6] = ht*hy/( 18.0*hx);  A_space_x[5][7] =  -ht*hy/( 18.0*hx);

    A_space_x[6][0] =  -ht*hy/(36.0*hx);  A_space_x[6][1] = ht*hy/(36.0*hx);  A_space_x[6][2] = ht*hy/(18.0*hx);  A_space_x[6][3] =  -ht*hy/(18.0*hx);
    A_space_x[6][4] =  -ht*hy/(18.0*hx);  A_space_x[6][5] = ht*hy/(18.0*hx);  A_space_x[6][6] = ht*hy/( 9.0*hx);  A_space_x[6][7] =  -ht*hy/( 9.0*hx);

    A_space_x[7][0] =  ht*hy/(36.0*hx);  A_space_x[7][1] = -ht*hy/(36.0*hx);  A_space_x[7][2] = -ht*hy/(18.0*hx);  A_space_x[7][3] =  ht*hy/(18.0*hx);
    A_space_x[7][4] =  ht*hy/(18.0*hx);  A_space_x[7][5] = -ht*hy/(18.0*hx);  A_space_x[7][6] = -ht*hy/( 9.0*hx);  A_space_x[7][7] =  ht*hy/( 9.0*hx);

    /* y-derivative term */
    A_space_y[0][0] =  ht*hx/( 9.0*hy);  A_space_y[0][1] =  ht*hx/(18.0*hy);  A_space_y[0][2] = -ht*hx/(18.0*hy);  A_space_y[0][3] = -ht*hx/( 9.0*hy);
    A_space_y[0][4] =  ht*hx/(18.0*hy);  A_space_y[0][5] =  ht*hx/(36.0*hy);  A_space_y[0][6] = -ht*hx/(36.0*hy);  A_space_y[0][7] = -ht*hx/(18.0*hy);

    A_space_y[1][0] =  ht*hx/(18.0*hy);  A_space_y[1][1] =  ht*hx/( 9.0*hy);  A_space_y[1][2] = -ht*hx/( 9.0*hy);  A_space_y[1][3] = -ht*hx/(18.0*hy);
    A_space_y[1][4] =  ht*hx/(36.0*hy);  A_space_y[1][5] =  ht*hx/(18.0*hy);  A_space_y[1][6] = -ht*hx/(18.0*hy);  A_space_y[1][7] = -ht*hx/(36.0*hy);

    A_space_y[2][0] = -ht*hx/(18.0*hy);  A_space_y[2][1] = -ht*hx/( 9.0*hy);  A_space_y[2][2] =  ht*hx/( 9.0*hy);  A_space_y[2][3] =  ht*hx/(18.0*hy);
    A_space_y[2][4] = -ht*hx/(36.0*hy);  A_space_y[2][5] = -ht*hx/(18.0*hy);  A_space_y[2][6] =  ht*hx/(18.0*hy);  A_space_y[2][7] =  ht*hx/(36.0*hy);

    A_space_y[3][0] = -ht*hx/( 9.0*hy);  A_space_y[3][1] = -ht*hx/(18.0*hy);  A_space_y[3][2] =  ht*hx/(18.0*hy);  A_space_y[3][3] =  ht*hx/( 9.0*hy);
    A_space_y[3][4] = -ht*hx/(18.0*hy);  A_space_y[3][5] = -ht*hx/(36.0*hy);  A_space_y[3][6] =  ht*hx/(36.0*hy);  A_space_y[3][7] =  ht*hx/(18.0*hy);

    A_space_y[4][0] =  ht*hx/(18.0*hy);  A_space_y[4][1] =  ht*hx/(36.0*hy);  A_space_y[4][2] = -ht*hx/(36.0*hy);  A_space_y[4][3] = -ht*hx/(18.0*hy);
    A_space_y[4][4] =  ht*hx/( 9.0*hy);  A_space_y[4][5] =  ht*hx/(18.0*hy);  A_space_y[4][6] = -ht*hx/(18.0*hy);  A_space_y[4][7] = -ht*hx/( 9.0*hy);

    A_space_y[5][0] =  ht*hx/(36.0*hy);  A_space_y[5][1] =  ht*hx/(18.0*hy);  A_space_y[5][2] = -ht*hx/(18.0*hy);  A_space_y[5][3] = -ht*hx/(36.0*hy);
    A_space_y[5][4] =  ht*hx/(18.0*hy);  A_space_y[5][5] =  ht*hx/( 9.0*hy);  A_space_y[5][6] = -ht*hx/( 9.0*hy);  A_space_y[5][7] = -ht*hx/(18.0*hy);

    A_space_y[6][0] = -ht*hx/(36.0*hy);  A_space_y[6][1] = -ht*hx/(18.0*hy);  A_space_y[6][2] =  ht*hx/(18.0*hy);  A_space_y[6][3] =  ht*hx/(36.0*hy);
    A_space_y[6][4] = -ht*hx/(18.0*hy);  A_space_y[6][5] = -ht*hx/( 9.0*hy);  A_space_y[6][6] =  ht*hx/( 9.0*hy);  A_space_y[6][7] =  ht*hx/(18.0*hy);

    A_space_y[7][0] = -ht*hx/(18.0*hy);  A_space_y[7][1] = -ht*hx/(36.0*hy);  A_space_y[7][2] =  ht*hx/(36.0*hy);  A_space_y[7][3] =  ht*hx/(18.0*hy);
    A_space_y[7][4] = -ht*hx/( 9.0*hy);  A_space_y[7][5] = -ht*hx/(18.0*hy);  A_space_y[7][6] =  ht*hx/(18.0*hy);  A_space_y[7][7] =  ht*hx/( 9.0*hy);

    /* time term */
    A_time[0][0] = -hx*hy/(18.0);  A_time[0][1] = -hx*hy/(36.0);  A_time[0][2] = -hx*hy/(72.0);  A_time[0][3] = -hx*hy/(36.0);
    A_time[0][4] =  hx*hy/(18.0);  A_time[0][5] =  hx*hy/(36.0);  A_time[0][6] =  hx*hy/(72.0);  A_time[0][7] =  hx*hy/(36.0);

    A_time[1][0] = -hx*hy/(36.0);  A_time[1][1] = -hx*hy/(18.0);  A_time[1][2] = -hx*hy/(36.0);  A_time[1][3] = -hx*hy/(72.0);
    A_time[1][4] =  hx*hy/(36.0);  A_time[1][5] =  hx*hy/(18.0);  A_time[1][6] =  hx*hy/(36.0);  A_time[1][7] =  hx*hy/(72.0);

    A_time[2][0] = -hx*hy/(72.0);  A_time[2][1] = -hx*hy/(36.0);  A_time[2][2] = -hx*hy/(18.0);  A_time[2][3] = -hx*hy/(36.0);
    A_time[2][4] =  hx*hy/(72.0);  A_time[2][5] =  hx*hy/(36.0);  A_time[2][6] =  hx*hy/(18.0);  A_time[2][7] =  hx*hy/(36.0);

    A_time[3][0] = -hx*hy/(36.0);  A_time[3][1] = -hx*hy/(72.0);  A_time[3][2] = -hx*hy/(36.0);  A_time[3][3] = -hx*hy/(18.0);
    A_time[3][4] =  hx*hy/(36.0);  A_time[3][5] =  hx*hy/(72.0);  A_time[3][6] =  hx*hy/(36.0);  A_time[3][7] =  hx*hy/(18.0);

    A_time[4][0] = -hx*hy/(18.0);  A_time[4][1] = -hx*hy/(36.0);  A_time[4][2] = -hx*hy/(72.0);  A_time[4][3] = -hx*hy/(36.0);
    A_time[4][4] =  hx*hy/(18.0);  A_time[4][5] =  hx*hy/(36.0);  A_time[4][6] =  hx*hy/(72.0);  A_time[4][7] =  hx*hy/(36.0);

    A_time[5][0] = -hx*hy/(36.0);  A_time[5][1] = -hx*hy/(18.0);  A_time[5][2] = -hx*hy/(36.0);  A_time[5][3] = -hx*hy/(72.0);
    A_time[5][4] =  hx*hy/(36.0);  A_time[5][5] =  hx*hy/(18.0);  A_time[5][6] =  hx*hy/(36.0);  A_time[5][7] =  hx*hy/(72.0);

    A_time[6][0] = -hx*hy/(72.0);  A_time[6][1] = -hx*hy/(36.0);  A_time[6][2] = -hx*hy/(18.0);  A_time[6][3] = -hx*hy/(36.0);
    A_time[6][4] =  hx*hy/(72.0);  A_time[6][5] =  hx*hy/(36.0);  A_time[6][6] =  hx*hy/(18.0);  A_time[6][7] =  hx*hy/(36.0);

    A_time[7][0] = -hx*hy/(36.0);  A_time[7][1] = -hx*hy/(72.0);  A_time[7][2] = -hx*hy/(36.0);  A_time[7][3] = -hx*hy/(18.0);
    A_time[7][4] =  hx*hy/(36.0);  A_time[7][5] =  hx*hy/(72.0);  A_time[7][6] =  hx*hy/(36.0);  A_time[7][7] =  hx*hy/(18.0);

    /* mass term */
    A_mass[0][0] = ht*hx*hy/( 27.0);  A_mass[0][1] = ht*hx*hy/( 54.0);  A_mass[0][2] = ht*hx*hy/(108.0);  A_mass[0][3] = ht*hx*hy/( 54.0);
    A_mass[0][4] = ht*hx*hy/( 54.0);  A_mass[0][5] = ht*hx*hy/(108.0);  A_mass[0][6] = ht*hx*hy/(216.0);  A_mass[0][7] = ht*hx*hy/(108.0);

    A_mass[1][0] = ht*hx*hy/( 54.0);  A_mass[1][1] = ht*hx*hy/( 27.0);  A_mass[1][2] = ht*hx*hy/( 54.0);  A_mass[1][3] = ht*hx*hy/(108.0);
    A_mass[1][4] = ht*hx*hy/(108.0);  A_mass[1][5] = ht*hx*hy/( 54.0);  A_mass[1][6] = ht*hx*hy/(108.0);  A_mass[1][7] = ht*hx*hy/(216.0);

    A_mass[2][0] = ht*hx*hy/(108.0);  A_mass[2][1] = ht*hx*hy/( 54.0);  A_mass[2][2] = ht*hx*hy/( 27.0);  A_mass[2][3] = ht*hx*hy/( 54.0);
    A_mass[2][4] = ht*hx*hy/(216.0);  A_mass[2][5] = ht*hx*hy/(108.0);  A_mass[2][6] = ht*hx*hy/( 54.0);  A_mass[2][7] = ht*hx*hy/(108.0);

    A_mass[3][0] = ht*hx*hy/( 54.0);  A_mass[3][1] = ht*hx*hy/(108.0);  A_mass[3][2] = ht*hx*hy/( 54.0);  A_mass[3][3] = ht*hx*hy/( 27.0);
    A_mass[3][4] = ht*hx*hy/(108.0);  A_mass[3][5] = ht*hx*hy/(216.0);  A_mass[3][6] = ht*hx*hy/(108.0);  A_mass[3][7] = ht*hx*hy/( 54.0);

    A_mass[4][0] = ht*hx*hy/( 54.0);  A_mass[4][1] = ht*hx*hy/(108.0);  A_mass[4][2] = ht*hx*hy/(216.0);  A_mass[4][3] = ht*hx*hy/(108.0);
    A_mass[4][4] = ht*hx*hy/( 27.0);  A_mass[4][5] = ht*hx*hy/( 54.0);  A_mass[4][6] = ht*hx*hy/(108.0);  A_mass[4][7] = ht*hx*hy/( 54.0);

    A_mass[5][0] = ht*hx*hy/(108.0);  A_mass[5][1] = ht*hx*hy/( 54.0);  A_mass[5][2] = ht*hx*hy/(108.0);  A_mass[5][3] = ht*hx*hy/(216.0);
    A_mass[5][4] = ht*hx*hy/( 54.0);  A_mass[5][5] = ht*hx*hy/( 27.0);  A_mass[5][6] = ht*hx*hy/( 54.0);  A_mass[5][7] = ht*hx*hy/(108.0);

    A_mass[6][0] = ht*hx*hy/(216.0);  A_mass[6][1] = ht*hx*hy/(108.0);  A_mass[6][2] = ht*hx*hy/( 54.0);  A_mass[6][3] = ht*hx*hy/(108.0);
    A_mass[6][4] = ht*hx*hy/(108.0);  A_mass[6][5] = ht*hx*hy/( 54.0);  A_mass[6][6] = ht*hx*hy/( 27.0);  A_mass[6][7] = ht*hx*hy/( 54.0);

    A_mass[7][0] = ht*hx*hy/(108.0);  A_mass[7][1] = ht*hx*hy/(216.0);  A_mass[7][2] = ht*hx*hy/(108.0);  A_mass[7][3] = ht*hx*hy/( 54.0);
    A_mass[7][4] = ht*hx*hy/( 54.0);  A_mass[7][5] = ht*hx*hy/(108.0);  A_mass[7][6] = ht*hx*hy/( 54.0);  A_mass[7][7] = ht*hx*hy/( 27.0);
}