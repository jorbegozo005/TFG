#include "nbodyJorge.h"
/*
extern int nplanetas;
extern int xyz;
extern int tamano;
extern int dimensiones;
extern int N;
extern int gmax;*/

__global__
void NbodyODE2Tpgpuaux(int nplanetas, int xyz, int tamano, int dimensiones, int N, float *Gm, float *aux, float *q, float *ddq) {

    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = threadIdx.x;
    int j = threadIdx.y;

    float Gmi, Gmj;

    if (i < nplanetas && j < nplanetas && i < j) {
        Gmi = Gm[i];
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
    
    /*for (int i=0; i<nplanetas; i++) {
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
    }*/
}

void NbodyODE2Tpgpu(int nplanetas, int xyz, int tamano, int dimensiones, int N, int gmax, float *ddq, float *q, struct NBodyParams *p) {

    for (int i=0; i<tamano*gmax; i++) {
        ddq[i] = 0.0;
    }
    float *Gm = p->gm;
    float *aux = p->aux;

    float *d_Gm, *d_aux, *d_q, *d_ddq;
    cudaMalloc(&d_Gm, nplanetas*sizeof(float));
    cudaMalloc(&d_aux, N*xyz*sizeof(float));
    cudaMalloc(&d_q, tamano*gmax*sizeof(float));
    cudaMalloc(&d_ddq, tamano*gmax*sizeof(float));

    cudaMemcpy(d_Gm, Gm, nplanetas*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, aux, N*xyz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ddq, ddq, tamano*gmax*sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (nplanetas + threadsPerBlock - 1) / threadsPerBlock;
    NbodyODE2Tpgpuaux<<<blocksPerGrid, threadsPerBlock>>>(nplanetas, xyz, tamano, dimensiones, N, d_Gm, d_aux, d_q, d_ddq);

    cudaMemcpy(aux, d_aux, N*xyz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ddq, d_ddq, tamano*gmax*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Gm);
    cudaFree(d_aux);
    cudaFree(d_q);
    cudaFree(d_ddq);

}