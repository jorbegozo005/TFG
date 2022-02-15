#include <stdlib.h>
#include <stdio.h>
#include "nbodyJorge.h"
#include "nbodyJorgegpu.h"

extern int nplanetas;
extern int xyz;
extern int tamano;
extern int dimensiones;
extern int N;
extern int gmax;

void TaylorLortuP(float *nu, float *nddu, struct NBodyParams *p) {
    int gradua = gmax-1;
    for (int i = 0; i<gradua/2; i++) {
        //NbodyODE2Tp(nddu,nu,p);
        NbodyODE2Tpgpu(nplanetas,xyz,tamano,dimensiones,N,gmax,nddu,nu,p);

        int k = 2*(i+1)+1;

        if (k < gmax) {   
            float zat=(k-1)*(k-2);
            float zat2=k*(k-1);
            dimensiones+=2;
            //nu = (float *) realloc((void *)nu, tamano*dimensiones*sizeof(float));

            for (int body=0; body<nplanetas; body++) {
                for (int koor=0; koor<xyz; koor++) {
                    nu[(k-1)*tamano+koor*nplanetas+body] =nddu[(k-3)*tamano+koor*nplanetas+body]/zat;
                    nu[k*tamano+koor*nplanetas+body] =nddu[(k-2)*tamano+koor*nplanetas+body]/zat2;
                }
            }
        } else {
            float zat=(k-1)*(k-2);
            dimensiones++;
            //nu = (float *) realloc(nu, tamano*dimensiones*sizeof(float));

            for (int body=0; body<nplanetas; body++) {
                for (int koor=0; koor<xyz; koor++) {
                    nu[(k-1)*tamano+koor*nplanetas+body] = nddu[(k-3)*tamano+koor*nplanetas+body]/zat;
                }
            }
        }       

    }
}

void evaluatetaylorv(float *nu, float h) {

    float x, y, z, dx, dy, dz, ukx, uky, ukz;

    for (int body=0; body<nplanetas; body++) {//body in 1:N
        x = nu[(dimensiones-1)*tamano+0+body*xyz]*h;
        nu[(dimensiones-1)*tamano+0+body*xyz]=0.0;
        y= nu[(dimensiones-1)*tamano+1+body*xyz]*h;
        nu[(dimensiones-1)*tamano+1+body*xyz]=0.0;
        z= nu[(dimensiones-1)*tamano+2+body*xyz]*h;
        nu[(dimensiones-1)*tamano+2+body*xyz]=0.0;
        dx= (dimensiones-1)*x;
        dy= (dimensiones-1)*y;
        dz= (dimensiones-1)*z;
        /*printf("EVALUATE\n");
        printf("x[%d]: %g\n", (dimensiones-1)*tamano+0+body*xyz, x);
        printf("y[%d]: %g\n", (dimensiones-1)*tamano+1+body*xyz, y);
        printf("z[%d]: %g\n", (dimensiones-1)*tamano+2+body*xyz, z);*/
        for (int k=dimensiones-1; k>1; k--) {//k in n-1:-1:3 // azken bi terminoak era berezian tratatuko ditut,
                          // izan ere, u'(t) espresioak ez baitauka t biderkatzen.
            ukx= nu[k*tamano+0+body*xyz];
            nu[k*tamano+0+body*xyz]=0.0;
            uky= nu[k*tamano+1+body*xyz];
            nu[k*tamano+1+body*xyz]=0.0;
            ukz= nu[k*tamano+2+body*xyz];
            nu[k*tamano+2+body*xyz]=0.0;
            x+=ukx;
            x*=h;
            y+=uky;
            y*=h;
            z+=ukz;
            z*=h;
            dx+= (k-1)*ukx;
            dx*=h;
            dy+= (k-1)*uky;
            dy*=h;
            dz+= (k-1)*ukz;
            dz*=h;
        }
        // orain bigarren terminoari dagokiona: 
        // u(t) kasuan orain artekoa honi gehitu eta dena * h
        // u'(t) kasuan  u_2 balioari gehitu orain artekoa eta laga u_2 berri bezala
        ukx= nu[1*tamano+0+body*xyz];
        uky= nu[1*tamano+1+body*xyz];
        ukz= nu[1*tamano+2+body*xyz];
        x+=ukx;
        x*=h;
        y+=uky;
        y*=h;
        z+=ukz;
        z*=h;
        nu[1*tamano+0+body*xyz] += dx;
        nu[1*tamano+1+body*xyz] += dy;
        nu[1*tamano+2+body*xyz] += dz;
        // Eta bukatzeko, lehenengo terminoari dagokiona: hau u(t) espresioan bakarrik dago,
        // eta u_1 koefizienteari orain artekoa gehitu behar diot
        nu[/*0*tamano+0+*/body*xyz] += x;
        nu[1+body*xyz] += y;
        nu[2+body*xyz] += z;
    }
}

void TaylorStepP(float *u, float *ddu, struct NBodyParams *p, float h) {
    TaylorLortuP(u,ddu,p);    
    evaluatetaylorv(u,h);
}


void IntegrateTaylorP(float *u, int t0, int tf, float h, float *gm) {

    float *ddu;
    ddu = (float *) malloc(tamano*gmax*sizeof(float));
    for (int i=0; i<tamano*gmax; i++) {
        ddu[i] = 0.0;
    }

    float *aux;
    aux = (float *) malloc(N*xyz*sizeof(float));
    for (int i=0; i<N*xyz; i++) {
        aux[i] = 0.0;
    }

    struct NBodyParams *params;
    params = (struct NBodyParams *) malloc(sizeof(struct NBodyParams));
    params->gm = gm;
    params->aux = aux;

    printf("Inicializada u:\n");
    for (int i=0; i<gmax; i++){
        for (int j=0; j<nplanetas; j++) {
            printf("%g %g %g;   ", u[i*tamano+j*xyz], u[i*tamano+j*xyz+1], u[i*tamano+j*xyz+2]);
        }
        printf("\n");
    }

    int ukop = (tf-t0)/h;
    printf("%d\n", ukop);
    //ukop = 3;
    for (int i=1; i<=ukop; i++) {
        TaylorStepP(u, ddu, params, h);
        /*printf("Step\n");
        for (int i=0; i<gmax; i++){
            for (int j=0; j<nplanetas; j++) {
                printf("%g %g %g;   ", u[i*tamano+j*xyz], u[i*tamano+j*xyz+1], u[i*tamano+j*xyz+2]);
            }
            printf("\n");
        }*/
        dimensiones = 2;
    }
    printf("FINAL");
    for (int i=0; i<gmax; i++){
        for (int j=0; j<nplanetas; j++) {
            printf("%g %g %g;   ", u[i*tamano+j*xyz], u[i*tamano+j*xyz+1], u[i*tamano+j*xyz+2]);
        }
        printf("\n");
    }
}
