#include <stdio.h>
#include <math.h>

#define N 1024
#define TILE_WIDTH 32

void transposeCPU(float *in, float *out, int n)
{
  for (int row = 0; row < n; row++)
    for (int col = 0; col < n; col++)
      out[col * n + row] = in[row * n + col];
}

__global__ void transposeNaive(float *in, float *out, int n)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < n && col < n)
    out[col * n + row] = in[row * n + col];
}

__global__ void transposeOptimized(float *in, float *out, int n)
{
  __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

  if (row < n && col < n)
    tile[threadIdx.y][threadIdx.x] = in[row * n + col];

  __syncthreads();

  col = blockIdx.y * TILE_WIDTH + threadIdx.x;
  row = blockIdx.x * TILE_WIDTH + threadIdx.y;

  if (row < n && col < n)
    out[row * n + col] = tile[threadIdx.x][threadIdx.y];
}

void runExperiment(int n)
{
  size_t size = (size_t)n * n * sizeof(float);

  float *in, *out_naive, *out_opt, *out_cpu;
  cudaMallocManaged(&in, size);
  cudaMallocManaged(&out_naive, size);
  cudaMallocManaged(&out_opt, size);
  out_cpu = (float *)malloc(size);

  for (int i = 0; i < n * n; i++)
    in[i] = (float)(i % 100) / 100.0f;

  cudaMemLocation gpuLoc;
  gpuLoc.type = cudaMemLocationTypeDevice;
  gpuLoc.id = 0;
  cudaMemPrefetchAsync(in, size, gpuLoc, 0);
  cudaMemPrefetchAsync(out_naive, size, gpuLoc, 0);
  cudaMemPrefetchAsync(out_opt, size, gpuLoc, 0);
  cudaDeviceSynchronize();

  clock_t start = clock();
  transposeCPU(in, out_cpu, n);
  clock_t end = clock();
  double cpuMs = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  float milliseconds = 0;

  dim3 threads(TILE_WIDTH, TILE_WIDTH);
  dim3 blocks((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

  cudaEventRecord(evStart);
  transposeNaive<<<blocks, threads>>>(in, out_naive, n);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&milliseconds, evStart, evStop);
  float naiveMs = milliseconds;

  cudaEventRecord(evStart);
  transposeOptimized<<<blocks, threads>>>(in, out_opt, n);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&milliseconds, evStart, evStop);
  float optMs = milliseconds;

  bool error = false;
  for (int i = 0; i < n * n && !error; i++)
  {
    if (fabsf(out_cpu[i] - out_opt[i]) > 1e-5f)
    {
      printf("MISMATCH at index %d\n", i);
      error = true;
    }
  }

  printf("\nMatrix size: %d x %d\n", n, n);
  printf("CPU time:               %8.3f ms\n", cpuMs);
  printf("Naive GPU time:         %8.3f ms  (speedup vs CPU: %.2fx)\n", naiveMs, cpuMs / naiveMs);
  printf("Optimized GPU time:     %8.3f ms  (speedup vs CPU: %.2fx, vs naive: %.2fx)\n", optMs, cpuMs / optMs, naiveMs / optMs);
  printf("Correctness: %s\n", error ? "MISMATCH" : "OK");

  free(out_cpu);
  cudaFree(in);
  cudaFree(out_naive);
  cudaFree(out_opt);
  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);
}

int main()
{
  int sizes[] = {512, 1024, 2048};
  for (int i = 0; i < 3; i++)
    runExperiment(sizes[i]);

  return 0;
}