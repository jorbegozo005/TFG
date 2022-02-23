#include <stdio.h>
#include <time.h>

__global__
void prueba(int n, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i];
}

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  printf("Bloque = %d, Thread = %d\n", blockIdx.x, threadIdx.x);
  
  if (i < n) {
    y[i] = a*x[i] + y[i];
  }
}

void saxpycpu(int n, float a, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = a*x[i] + y[i];
    //printf("Eso %f\n", y[i]);
  }
}

int main(void)
{
  
  int N = 5;//1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    printf("%d ",i);
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  clock_t begin = clock();

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  for (int i=0; i<5; i++) {

    printf("Antes\n");

    //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
    saxpy<<<5,3>>>(N, 2.0f, d_x, d_y);

    cudaDeviceSynchronize();
    printf("Sinc\n");
  }
  //saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  //cudaDeviceSynchronize();

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  printf("Tiempo gpu: %f\n", time_spent);

  for (int i = 0; i < N; i++) {
    printf("%d ",i);
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  begin = clock();

  for (int i=0; i<5; i++) {

    saxpycpu(N, 2.0f, x, y);
    
  }

  //saxpycpu(N, 2.0f, x, y);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  printf("Tiempo cpu: %f\n", time_spent);


  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);


}