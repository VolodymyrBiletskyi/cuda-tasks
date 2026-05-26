#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define MAX_KERNEL_SIZE 49

__constant__ float constKernel[MAX_KERNEL_SIZE];

void convolveCPU(float *input, float *output, float *kernel,
                 int width, int height, int kSize, int boundaryMode)
{
  int radius = kSize / 2;
  for (int row = 0; row < height; row++)
  {
    for (int col = 0; col < width; col++)
    {
      float sum = 0.0f;
      for (int kr = -radius; kr <= radius; kr++)
      {
        for (int kc = -radius; kc <= radius; kc++)
        {
          int r = row + kr;
          int c = col + kc;
          float pixel = 0.0f;
          if (boundaryMode == 0)
          {
            if (r >= 0 && r < height && c >= 0 && c < width)
              pixel = input[r * width + c];
          }
          else
          {
            r = r < 0 ? 0 : (r >= height ? height - 1 : r);
            c = c < 0 ? 0 : (c >= width ? width - 1 : c);
            pixel = input[r * width + c];
          }
          sum += pixel * kernel[(kr + radius) * kSize + (kc + radius)];
        }
      }
      output[row * width + col] = sum;
    }
  }
}

__global__ void convolveNaive(float *input, float *output,
                              int width, int height, int kSize)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width)
  {
    int radius = kSize / 2;
    float sum = 0.0f;
    for (int kr = -radius; kr <= radius; kr++)
    {
      for (int kc = -radius; kc <= radius; kc++)
      {
        int r = row + kr;
        int c = col + kc;
        float pixel = 0.0f;
        if (r >= 0 && r < height && c >= 0 && c < width)
          pixel = input[r * width + c];
        sum += pixel * constKernel[(kr + radius) * kSize + (kc + radius)];
      }
    }
    output[row * width + col] = sum;
  }
}

__global__ void convolveTiled(float *input, float *output,
                              int width, int height, int kSize, int tileSize)
{
  extern __shared__ float shInput[];

  int radius = kSize / 2;
  int shWidth = tileSize + kSize - 1;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * tileSize + tx;
  int row = blockIdx.y * tileSize + ty;

  for (int dr = ty; dr < shWidth; dr += tileSize)
  {
    for (int dc = tx; dc < shWidth; dc += tileSize)
    {
      int r = blockIdx.y * tileSize + dr - radius;
      int c = blockIdx.x * tileSize + dc - radius;
      float pixel = 0.0f;
      if (r >= 0 && r < height && c >= 0 && c < width)
        pixel = input[r * width + c];
      shInput[dr * shWidth + dc] = pixel;
    }
  }

  __syncthreads();

  if (row < height && col < width)
  {
    float sum = 0.0f;
    for (int kr = 0; kr < kSize; kr++)
      for (int kc = 0; kc < kSize; kc++)
        sum += shInput[(ty + kr) * shWidth + (tx + kc)] * constKernel[kr * kSize + kc];
    output[row * width + col] = sum;
  }
}

void makeGaussianKernel(float *kernel, int kSize, float sigma)
{
  int radius = kSize / 2;
  float sum = 0.0f;
  for (int kr = -radius; kr <= radius; kr++)
  {
    for (int kc = -radius; kc <= radius; kc++)
    {
      float val = expf(-(kr * kr + kc * kc) / (2.0f * sigma * sigma));
      kernel[(kr + radius) * kSize + (kc + radius)] = val;
      sum += val;
    }
  }
  for (int i = 0; i < kSize * kSize; i++)
    kernel[i] /= sum;
}

void runExperiment(int kSize, int tileSize)
{
  int width = IMAGE_WIDTH, height = IMAGE_HEIGHT;
  size_t imgSize = (size_t)width * height * sizeof(float);
  size_t krnSize = (size_t)kSize * kSize * sizeof(float);

  float *input, *output_cpu, *output_naive, *output_tiled;
  cudaMallocManaged(&input, imgSize);
  cudaMallocManaged(&output_cpu, imgSize);
  cudaMallocManaged(&output_naive, imgSize);
  cudaMallocManaged(&output_tiled, imgSize);

  float *kernel = (float *)malloc(krnSize);

  for (int i = 0; i < width * height; i++)
    input[i] = (float)(i % 256) / 255.0f;

  makeGaussianKernel(kernel, kSize, 1.0f);

  cudaMemcpyToSymbol(constKernel, kernel, krnSize);

  cudaMemLocation gpuLoc;
  gpuLoc.type = cudaMemLocationTypeDevice;
  gpuLoc.id = 0;
  cudaMemPrefetchAsync(input, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_naive, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_tiled, imgSize, gpuLoc, 0);
  cudaDeviceSynchronize();

  clock_t start = clock();
  convolveCPU(input, output_cpu, kernel, width, height, kSize, 0);
  clock_t end = clock();
  double cpuMs = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  float naiveMs = 0, tiledMs = 0;

  dim3 naiveThreads(16, 16);
  dim3 naiveBlocks((width + 15) / 16, (height + 15) / 16);

  cudaEventRecord(evStart);
  convolveNaive<<<naiveBlocks, naiveThreads>>>(input, output_naive,
                                               width, height, kSize);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&naiveMs, evStart, evStop);

  dim3 tiledThreads(tileSize, tileSize);
  dim3 tiledBlocks((width + tileSize - 1) / tileSize,
                   (height + tileSize - 1) / tileSize);
  int shWidth = tileSize + kSize - 1;
  size_t shMemSize = shWidth * shWidth * sizeof(float);

  cudaEventRecord(evStart);
  convolveTiled<<<tiledBlocks, tiledThreads, shMemSize>>>(input, output_tiled,
                                                          width, height, kSize, tileSize);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&tiledMs, evStart, evStop);

  bool error = false;
  for (int i = 0; i < width * height && !error; i++)
  {
    if (fabsf(output_cpu[i] - output_tiled[i]) > 1e-3f)
    {
      printf("MISMATCH at %d: cpu=%.6f tiled=%.6f\n", i, output_cpu[i], output_tiled[i]);
      error = true;
    }
  }

  printf("Kernel %dx%d | Tile %dx%d | CPU: %.3f ms | Naive+Const: %.3f ms | Tiled+Const: %.3f ms | Speedup vs naive: %.2fx | %s\n",
         kSize, kSize, tileSize, tileSize, cpuMs, naiveMs, tiledMs,
         naiveMs / tiledMs, error ? "MISMATCH" : "OK");

  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);
  free(kernel);
  cudaFree(input);
  cudaFree(output_cpu);
  cudaFree(output_naive);
  cudaFree(output_tiled);
}

int main()
{
  int kSizes[] = {3, 5, 7};
  int tileSizes[] = {8, 16, 32};

  for (int s = 0; s < 3; s++)
    for (int t = 0; t < 3; t++)
      runExperiment(kSizes[s], tileSizes[t]);

  return 0;
}