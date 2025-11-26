#include <stdlib.h>
#include "appctx.h"
#include "nonlin.h"

PetscErrorCode FormJacobian(SNES snes, Vec U, Mat J, Mat P, void *ctx)
{
    MatSetOption(J, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
    if (P != J) { MatSetOption(P, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE); }
    MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);

    AppCtx           *user = (AppCtx*)ctx;
    DM                dm   = user->dm;
    DMDALocalInfo     info;
    MatStencil        row[32], col[32];
    PetscInt          xs,ys,zs,xm,ym,zm, nx,ny,nt;
    PetscReal         hx,hy,ht;
    PetscScalar     (***u)[4];

    PetscFunctionBegin;
    nx = user->nx;  ny = user->ny; nt = user->nt;
    hx = user->hx;  hy = user->hy; ht = user->ht;

    DMDAGetLocalInfo(dm,&info);
    xs = info.xs; xm = info.xm;
    ys = info.ys; ym = info.ym;
    zs = info.zs; zm = info.zm;

    MatZeroEntries(J);
    if (P != J) MatZeroEntries(P);

    PetscScalar (*A_time)[8]    = user->A_time;
    PetscScalar (*A_space_x)[8] = user->A_space_x;
    PetscScalar (*A_space_y)[8] = user->A_space_y;
    PetscScalar (*A_mass)[8]    = user->A_mass;

    Vec Ul;
    DMGetLocalVector(dm,&Ul);
    DMGlobalToLocalBegin(dm,U,INSERT_VALUES,Ul);
    DMGlobalToLocalEnd  (dm,U,INSERT_VALUES,Ul);
    DMDAVecGetArrayRead(dm,Ul,&u);

    PetscInt tEnd = PetscMin(zs + zm - 1, nt - 2);
    for (PetscInt tElm = zs; tElm <= tEnd; ++tElm) {
        for (PetscInt yElm = ys; yElm < ys+ym;   ++yElm) {
            for (PetscInt xElm = xs; xElm < xs+xm;   ++xElm) {

                PetscInt xloc0 = xElm;
                PetscInt xloc1 = (xElm + 1);
                PetscInt yloc0 = yElm;
                PetscInt yloc1 = (yElm + 1);
                PetscInt xg0   = xElm;
                PetscInt xg1   = (xElm + 1);
                PetscInt yg0   = yElm;
                PetscInt yg1   = (yElm + 1);
                PetscInt t0    = tElm;
                PetscInt t1    = tElm + 1;

                PetscInt xglob[8] = { xg0, xg1, xg1, xg0, xg0, xg1, xg1, xg0 };
                PetscInt yglob[8] = { yg0, yg0, yg1, yg1, yg0, yg0, yg1, yg1 };
                PetscInt tglob[8] = {  t0,  t0,  t0,  t0,  t1,  t1,  t1,  t1 };
                int k = 0;
                for (int corner = 0; corner < 8; ++corner) {
                    for (int comp = 0; comp < 4; ++comp, ++k) {
                        row[k].i = xglob[corner];
                        row[k].j = yglob[corner];
                        row[k].k = tglob[corner];
                        row[k].c = comp;
                        col[k]   = row[k];
                    }
                }

                PetscScalar phi_l[8], chi_l[8];

                phi_l[0]=u[t0][yloc0][xloc0][0];
                phi_l[1]=u[t0][yloc0][xloc1][0];
                phi_l[2]=u[t0][yloc1][xloc1][0];
                phi_l[3]=u[t0][yloc1][xloc0][0];
                phi_l[4]=u[t1][yloc0][xloc0][0];
                phi_l[5]=u[t1][yloc0][xloc1][0];
                phi_l[6]=u[t1][yloc1][xloc1][0];
                phi_l[7]=u[t1][yloc1][xloc0][0];

                chi_l[0]=u[t0][yloc0][xloc0][2];
                chi_l[1]=u[t0][yloc0][xloc1][2];
                chi_l[2]=u[t0][yloc1][xloc1][2];
                chi_l[3]=u[t0][yloc1][xloc0][2];
                chi_l[4]=u[t1][yloc0][xloc0][2];
                chi_l[5]=u[t1][yloc0][xloc1][2];
                chi_l[6]=u[t1][yloc1][xloc1][2];
                chi_l[7]=u[t1][yloc1][xloc0][2];

                PetscScalar M[32][32] = {{0}};
                for (int i=0;i<8;++i){
                    int r1 = 4*i, r2=r1+1, r3=r1+2, r4=r1+3;
                    for (int j=0;j<8;++j){
                        int cphi = 4*j,  cu=cphi+1,  cchi=cphi+2,  cv=cphi+3;
                        M[r1][cphi] += (A_space_x[i][j] + A_space_y[i][j]) + user->mphi2*A_mass[i][j];
                        M[r1][cu]   += A_time [i][j];
                        M[r2][cphi] += A_time [i][j];
                        M[r2][cu]   += -A_mass [i][j];

                        M[r3][cchi] += (A_space_x[i][j] + A_space_y[i][j]) + user->mchi2*A_mass[i][j];
                        M[r3][cv]   += A_time [i][j];
                        M[r4][cchi] += A_time [i][j];
                        M[r4][cv]   += -A_mass [i][j];
                    }
                }

                PetscScalar Jpp[8][8], Jpchi[8][8], Jchichi[8][8], Jchiphi[8][8];
                ComputeJ_local_nonlinear(Jpp,Jpchi,Jchichi,Jchiphi,phi_l,chi_l,hx,hy,ht,ctx);
                for (int i=0;i<8;++i){
                    int r1=4*i, r3=r1+2;
                    for (int j=0;j<8;++j){
                        int cphi=4*j, cchi=cphi+2;
                        M[r1][cphi]+=Jpp[i][j];     M[r1][cchi]+=Jpchi[i][j];
                        M[r3][cchi]+=Jchichi[i][j]; M[r3][cphi]+=Jchiphi[i][j];
                    }
                }

                MatSetValuesStencil(J,32,row,32,col,&M[0][0],ADD_VALUES);
                if (P!=J) MatSetValuesStencil(P,32,row,32,col,&M[0][0],ADD_VALUES);
            }
        }
    }

    DMDAVecRestoreArrayRead(dm,Ul,&u);
    DMRestoreLocalVector(dm,&Ul);

    MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
    if (P!=J){MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);}

    PetscInt ndofPerNode = 4;
    PetscInt nlocal     = info.xm * info.ym * ndofPerNode;
    MatStencil *bc = (MatStencil*)malloc(nlocal * sizeof(*bc));
    if (!bc) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"malloc failed for bc");
    PetscInt idx = 0;
    for (PetscInt y = info.ys; y < info.ys + info.ym; ++y) {
        for (PetscInt x = info.xs; x < info.xs + info.xm; ++x) {
            for (int comp = 0; comp < ndofPerNode; ++comp, ++idx) {
                bc[idx].i = x;
                bc[idx].j = y;
                bc[idx].k = 0;
                bc[idx].c = comp;
            }
        }
    }

    MatZeroRowsStencil(J, nlocal, bc, 1.0, NULL, NULL);
    if (P != J) MatZeroRowsStencil(P, nlocal, bc, 1.0, NULL, NULL);

    free(bc);

    MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
    if (P!=J){MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);}

    PetscFunctionReturn(0);
}