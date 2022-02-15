#include <stdlib.h>
#include <time.h>
#include <stdio.h>
//#include "initialInnerPlanets.h"
//#include "taylorJorge.h"
//#include "nbodyJorge.h"

int nplanetas, tamano, dimensiones, N, xyz, gmax;
float *d_Gm, *d_aux, *d_q, *d_ddq;
bool gpu;

//#include "nbodyJorge.h"
/*
extern int nplanetas;
extern int xyz;
extern int tamano;
extern int dimensiones;
extern int N;
extern int gmax;*/

__global__
void NbodyODE2Tpgpuaux(int gmax, int nplanetas, int xyz, int tamano, int dimensiones, int N, float *Gm, float *aux, float *q, float *ddq) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //int i = threadIdx.x;
    //int j = threadIdx.y;

    float Gmi, Gmj;
    //float Gmi;

    if (i < nplanetas && j < nplanetas && i < j) {
    //if (i < nplanetas) {
    //for (int i=0; i<nplanetas; i++) {
        Gmi = Gm[i];
        //for (int j=i+1; j<nplanetas; j++) {
        //   float Gmj = Gm[j];
        //Gmi = Gm[i];
        Gmj = Gm[j];
        
            for (int k=0; k<dimensiones; k++) {
                aux[k*N+0] = q[k*tamano+0+i*xyz] - q[k*tamano+0+j*xyz];
                aux[k*N+1] = q[k*tamano+1+i*xyz] - q[k*tamano+1+j*xyz];
                aux[k*N+2] = q[k*tamano+2+i*xyz] - q[k*tamano+2+j*xyz];
            }
            for (int k=0; k<dimensiones; k++) {
                aux[k*N+3] = 0.0;
                aux[k*N+4] = 0.0;
                aux[k*N+5] = 0.0;
                for (int m=0; m<=k; m++) {
                    aux[k*N+3] += aux[m*N+0] * aux[(k-m)*N+0];
                    aux[k*N+4] += aux[m*N+1] * aux[(k-m)*N+1];
                    aux[k*N+5] += aux[m*N+2] * aux[(k-m)*N+2];
                }
            }

            for (int k=0; k<dimensiones; k++) {
                aux[k*N+6] = aux[k*N+3] + aux[k*N+4] + aux[k*N+5];
                aux[k*N+7] = 0.0;
            }

            float berretzailea = -3.0/2.0;
            aux[7] = pow(aux[6],berretzailea);
            for (int k=1; k<=dimensiones-1; k++) {
                float lag = 0.0;
                for (int m=0; m<=k; m++) {
                    lag = lag + (berretzailea*(k-m)-m)*aux[(k-1-m+1)*N+6]*aux[m*N+7];
                }
                aux[k*N+7]=lag/(k*aux[6]);
            }            

            for (int k=0; k<dimensiones; k++) {
                aux[k*N+3]=0.0;
                aux[k*N+4]=0.0;
                aux[k*N+5]=0.0;
                for (int m=1; m<=k+1; m++) {
                    aux[k*N+3] +=  aux[(m-1)*N+7]*aux[(k-m+1)*N];
                    aux[k*N+4] +=  aux[(m-1)*N+7]*aux[(k-m+1)*N+1];
                    aux[k*N+5] +=  aux[(m-1)*N+7]*aux[(k-m+1)*N+2];
                }
            }

            for (int k=0; k<dimensiones; k++) { 
                ddq[k*tamano+0+i*xyz] -= Gmj*aux[k*N+3];
                ddq[k*tamano+0+j*xyz] += Gmi*aux[k*N+3];
                ddq[k*tamano+1+i*xyz] -= Gmj*aux[k*N+4];
                ddq[k*tamano+1+j*xyz] += Gmi*aux[k*N+4];
                ddq[k*tamano+2+i*xyz] -= Gmj*aux[k*N+5];
                ddq[k*tamano+2+j*xyz] += Gmi*aux[k*N+5];
            }
        }
    //}
    
}

void NbodyODE2Tpgpu(float *ddq, float *q, float *Gm, float *aux) {
    
    for (int i=0; i<tamano*gmax; i++) {
        ddq[i] = 0.0;
    }
    
    /*cudaMemcpy(d_Gm, Gm, nplanetas*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, aux, N*xyz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ddq, ddq, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);*/

    //int threadsPerBlock = 256;
    //int blocksPerGrid = (nplanetas + threadsPerBlock - 1) / threadsPerBlock;
    
    NbodyODE2Tpgpuaux<<<(xyz*tamano*dimensiones)/32, 32>>>(gmax, nplanetas, xyz, tamano, dimensiones, N, d_Gm, d_aux, d_q, d_ddq);

    /*cudaMemcpy(Gm, d_Gm, nplanetas*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(aux, d_aux, N*xyz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(q, d_q, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ddq, d_ddq, tamano*gmax*sizeof(float), cudaMemcpyDeviceToHost);*/
}

void inicializarGlobales() {
    nplanetas = 5;
    xyz = 3;
    tamano = nplanetas*xyz;
    dimensiones = 2;
    N = 8;
    gmax = 8;
    gpu = true;
}

void NbodyODE2Tp(float *ddq, float *q, float *Gm, float *aux) {

    for (int i=0; i<tamano*gmax; i++) {
        ddq[i] = 0.0;
    }
    
    for (int i=0; i<nplanetas; i++) {
        float Gmi = Gm[i];
        for (int j=i+1; j<nplanetas; j++) {
           float Gmj = Gm[j];
           for (int k=0; k<dimensiones; k++) {
               aux[k*N+0] = q[k*tamano+0+i*xyz] - q[k*tamano+0+j*xyz];
               aux[k*N+1] = q[k*tamano+1+i*xyz] - q[k*tamano+1+j*xyz];
               aux[k*N+2] = q[k*tamano+2+i*xyz] - q[k*tamano+2+j*xyz];
           }
           for (int k=0; k<dimensiones; k++) {
                aux[k*N+3] = 0.0;
                aux[k*N+4] = 0.0;
                aux[k*N+5] = 0.0;
                for (int m=0; m<=k; m++) {
                    aux[k*N+3] += aux[m*N+0] * aux[(k-m)*N+0];
                    aux[k*N+4] += aux[m*N+1] * aux[(k-m)*N+1];
                    aux[k*N+5] += aux[m*N+2] * aux[(k-m)*N+2];
                }
           }

            for (int k=0; k<dimensiones; k++) {
                aux[k*N+6] = aux[k*N+3] + aux[k*N+4] + aux[k*N+5];
                aux[k*N+7] = 0.0;
            }


            float berretzailea = -3.0/2.0;
            aux[7] = pow(aux[6],berretzailea);
            for (int k=1; k<=dimensiones-1; k++) {
                float lag = 0.0;
                for (int m=0; m<=k; m++) {
                    lag = lag + (berretzailea*(k-m)-m)*aux[(k-1-m+1)*N+6]*aux[m*N+7];
                }
                aux[k*N+7]=lag/(k*aux[6]);
            }            

            for (int k=0; k<dimensiones; k++) {
                aux[k*N+3]=0.0;
                aux[k*N+4]=0.0;
                aux[k*N+5]=0.0;
                for (int m=1; m<=k+1; m++) {
                    aux[k*N+3] +=  aux[(m-1)*N+7]*aux[(k-m+1)*N];
                    aux[k*N+4] +=  aux[(m-1)*N+7]*aux[(k-m+1)*N+1];
                    aux[k*N+5] +=  aux[(m-1)*N+7]*aux[(k-m+1)*N+2];
                }
            }

            for (int k=0; k<dimensiones; k++) { 
                ddq[k*tamano+0+i*xyz] -= Gmj*aux[k*N+3];
                ddq[k*tamano+0+j*xyz] += Gmi*aux[k*N+3];
                ddq[k*tamano+1+i*xyz] -= Gmj*aux[k*N+4];
                ddq[k*tamano+1+j*xyz] += Gmi*aux[k*N+4];
                ddq[k*tamano+2+i*xyz] -= Gmj*aux[k*N+5];
                ddq[k*tamano+2+j*xyz] += Gmi*aux[k*N+5];
            }
        }
    }
}

void TaylorLortuP(float *nu, float *nddu, float *gm, float *aux) {
    int gradua = gmax-1;
    for (int i = 0; i<gradua/2; i++) {
        if (gpu) {
            NbodyODE2Tpgpu(nddu,nu,gm,aux);
        } else {
            NbodyODE2Tp(nddu,nu,gm,aux);
        }
        
        int k = 2*(i+1)+1;

        if (k < gmax) {   
            float zat=(k-1)*(k-2);
            float zat2=k*(k-1);
            dimensiones+=2;
            for (int body=0; body<nplanetas; body++) {
                for (int koor=0; koor<xyz; koor++) {
                    nu[(k-1)*tamano+koor*nplanetas+body] =nddu[(k-3)*tamano+koor*nplanetas+body]/zat;
                    nu[k*tamano+koor*nplanetas+body] =nddu[(k-2)*tamano+koor*nplanetas+body]/zat2;
                }
            }
        } else {
            float zat=(k-1)*(k-2);
            dimensiones++;
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

void TaylorStepP(float *u, float *ddu, float *gm, float *aux, float h) {
    TaylorLortuP(u,ddu,gm,aux);
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

    printf("Inicializada u:\n");
    for (int i=0; i<gmax; i++){
        for (int j=0; j<nplanetas; j++) {
            printf("%g %g %g;   ", u[i*tamano+j*xyz], u[i*tamano+j*xyz+1], u[i*tamano+j*xyz+2]);
        }
        printf("\n");
    }

    if (gpu) {
        cudaMalloc(&d_Gm, nplanetas*sizeof(float));
        cudaMalloc(&d_aux, N*xyz*sizeof(float));
        cudaMalloc(&d_q, tamano*gmax*sizeof(float));
        cudaMalloc(&d_ddq, tamano*gmax*sizeof(float));

        cudaMemcpy(d_Gm, gm, nplanetas*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_aux, aux, N*xyz*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, u, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ddq, ddu, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);
    }

    int ukop = (tf-t0)/h;
    printf("%d\n", ukop);
    ukop = 100000;
    clock_t begin = clock();
    for (int i=1; i<=ukop; i++) {
        if (gpu) {
            TaylorStepP(u, ddu, gm, aux, h);//TaylorStepP(d_q, d_ddq, d_Gm, d_aux, h);
        } else {
            TaylorStepP(u, ddu, gm, aux, h);
        }
        dimensiones = 2;
    }
    clock_t end = clock();

    if (gpu) {
        /*cudaMemcpy(gm, d_Gm, nplanetas*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(aux, d_aux, N*xyz*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(u, d_q, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ddu, d_ddq, tamano*gmax*sizeof(float), cudaMemcpyDeviceToHost);*/
    }

    printf("FINAL");
    for (int i=0; i<gmax; i++){
        for (int j=0; j<nplanetas; j++) {
            printf("%g %g %g;   ", u[i*tamano+j*xyz], u[i*tamano+j*xyz+1], u[i*tamano+j*xyz+2]);
        }
        printf("\n");
    }

    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Tiempo: %f\n", time_spent);
    
    if (gpu) {
        cudaFree(d_Gm);
        cudaFree(d_aux);
        cudaFree(d_q);
        cudaFree(d_ddq);
    }
    free(ddu);
    free(aux);
}

void initialInnerPlanets(float *Gm, float *u, int tamano) {

/*
Order of planets:
Sun, Mercury, Venus, Earth, Mars, Moon
*/

    float GmMoon = 0.109318945074237400e-10;
    float GmEarth = 0.888769244512563400e-9;

    Gm[0] = 0.295912208285591100e-3;
    Gm[1] = 0.491248045036476000e-10;
    Gm[2] = 0.724345233264412000e-9;
    Gm[3] = GmEarth+GmMoon;
    Gm[4] = 0.954954869555077000e-10;

    float q0[15] = {0.00450250878464055477, 0.00076707642709100705, 0.00026605791776697764,   // Sun
        0.36176271656028195477, -0.09078197215676599295, -0.08571497256275117236, // Mercury
        0.61275194083507215477, -0.34836536903362219295, -0.19527828667594382236, // Venus
        0.12051741410138465477, -0.92583847476914859295, -0.40154022645315222236, // EM bary
        -0.11018607714879824523, -1.32759945030298299295,-0.60588914048429142236  // Mars
    };

    float v[15] = {-0.00000035174953607552, 0.00000517762640983341, 0.00000222910217891203,  // Sun
      0.00336749397200575848, 0.02489452055768343341, 0.01294630040970409203,  // Mercury
      0.01095206842352823448, 0.01561768426786768341, 0.00633110570297786403,  // Venus
      0.01681126830978379448, 0.00174830923073434441, 0.00075820289738312913,  // EM bary
      0.01448165305704756448, 0.00024246307683646861, -0.00028152072792433877  // Mars
    };

    float sumGm = 0.0;
    for (int i=0; i<5; i++) {
       sumGm = sumGm + Gm[i];
    }

    float resultadoqx = 0.0;
    float resultadoqy = 0.0;
    float resultadoqz = 0.0;
    for (int i=0; i<5; i++) {
       resultadoqx =  resultadoqx + (Gm[i]*q0[i*3]);
       resultadoqy =  resultadoqy + (Gm[i]*q0[i*3+1]);
       resultadoqz =  resultadoqz + (Gm[i]*q0[i*3+2]);
    }
    float qxbar = resultadoqx/sumGm;
    float qybar = resultadoqy/sumGm;
    float qzbar = resultadoqz/sumGm;

    float resultadovx = 0.0;
    float resultadovy = 0.0;
    float resultadovz = 0.0;
    for (int i=0; i<5; i++) {
       resultadovx =  resultadovx + (Gm[i]*v[i*3]);
       resultadovy =  resultadovy + (Gm[i]*v[i*3+1]);
       resultadovz =  resultadovz + (Gm[i]*v[i*3+2]);
    }
    float vxbar = resultadovx/sumGm;
    float vybar = resultadovy/sumGm;
    float vzbar = resultadovz/sumGm;

    for (int i=0; i<5; i++) {
       q0[i*3] = q0[i*3] - qxbar;
       q0[i*3+1] = q0[i*3+1] - qybar;
       q0[i*3+2] = q0[i*3+2] - qzbar;
    }

    for (int i=0; i<5; i++) {
       v[i*3] = v[i*3] - vxbar;
       v[i*3+1] = v[i*3+1] - vybar;
       v[i*3+2] = v[i*3+2] - vzbar;
    }

    for (int i=0; i<tamano; i++) {
        u[i] = q0[i];
        u[i+tamano] = v[i];
    }

    for (int i=nplanetas*xyz*2; i<gmax*nplanetas*xyz; i++) {
        u[i] = 0.0;
    }

}


int main() {

    inicializarGlobales();

    float *GM;
    GM = (float *) malloc(nplanetas*sizeof(float));
    
    float *u;
    u = (float *) malloc(tamano*gmax*sizeof(float));

    initialInnerPlanets(GM, u, tamano);

    int t0 = 0;
    int tf = 360;
    float h = 0.01;

    IntegrateTaylorP(u, t0, tf, h, GM);

    free(GM);
    free(u);
    
}
