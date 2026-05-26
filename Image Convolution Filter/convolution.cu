#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define MAX_KERNEL_SIZE 49
#define MAX_KERNEL_1D 7
#define PADDING 1

__constant__ float constKernel[MAX_KERNEL_SIZE];
__constant__ float constKernel1D[MAX_KERNEL_1D];

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

__global__ void convolvePadded(float *input, float *output,
                               int width, int height, int kSize,
                               int tileSize, int pitch)
{
  extern __shared__ float shInput[];

  int radius = kSize / 2;
  int shWidth = tileSize + kSize - 1 + PADDING;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * tileSize + tx;
  int row = blockIdx.y * tileSize + ty;

  for (int dr = ty; dr < shWidth; dr += tileSize)
  {
    for (int dc = tx; dc < shWidth - PADDING; dc += tileSize)
    {
      int r = blockIdx.y * tileSize + dr - radius;
      int c = blockIdx.x * tileSize + dc - radius;
      float pixel = 0.0f;
      if (r >= 0 && r < height && c >= 0 && c < width)
        pixel = input[r * pitch + c];
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
    output[row * pitch + col] = sum;
  }
}

void makeGaussianKernel2D(float *kernel, int kSize, float sigma)
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

void makeGaussianKernel1D(float *kernel, int kSize, float sigma)
{
  int radius = kSize / 2;
  float sum = 0.0f;
  for (int k = -radius; k <= radius; k++)
  {
    float val = expf(-(k * k) / (2.0f * sigma * sigma));
    kernel[k + radius] = val;
    sum += val;
  }
  for (int i = 0; i < kSize; i++)
    kernel[i] /= sum;
}

void runExperiment(int kSize, int tileSize)
{
  int width = IMAGE_WIDTH, height = IMAGE_HEIGHT;
  int pitch = ((width + 31) / 32) * 32;
  size_t imgSize = (size_t)width * height * sizeof(float);
  size_t imgSizePadded = (size_t)pitch * height * sizeof(float);
  size_t krnSize2D = (size_t)kSize * kSize * sizeof(float);
  size_t krnSize1D = (size_t)kSize * sizeof(float);

  float *input, *input_padded, *output_cpu;
  float *output_tiled, *output_padded;
  cudaMallocManaged(&input, imgSize);
  cudaMallocManaged(&input_padded, imgSizePadded);
  cudaMallocManaged(&output_cpu, imgSize);
  cudaMallocManaged(&output_tiled, imgSize);
  cudaMallocManaged(&output_padded, imgSizePadded);

  float *kernel2D = (float *)malloc(krnSize2D);
  float *kernel1D = (float *)malloc(krnSize1D);

  for (int r = 0; r < height; r++)
    for (int c = 0; c < width; c++)
    {
      float val = (float)((r * width + c) % 256) / 255.0f;
      input[r * width + c] = val;
      input_padded[r * pitch + c] = val;
    }

  makeGaussianKernel2D(kernel2D, kSize, 1.0f);
  makeGaussianKernel1D(kernel1D, kSize, 1.0f);

  cudaMemcpyToSymbol(constKernel, kernel2D, krnSize2D);
  cudaMemcpyToSymbol(constKernel1D, kernel1D, krnSize1D);

  cudaMemLocation gpuLoc;
  gpuLoc.type = cudaMemLocationTypeDevice;
  gpuLoc.id = 0;
  cudaMemPrefetchAsync(input, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(input_padded, imgSizePadded, gpuLoc, 0);
  cudaMemPrefetchAsync(output_tiled, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_padded, imgSizePadded, gpuLoc, 0);
  cudaDeviceSynchronize();

  clock_t start = clock();
  convolveCPU(input, output_cpu, kernel2D, width, height, kSize, 0);
  clock_t end = clock();
  double cpuMs = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  float tiledMs = 0, paddedMs = 0;

  dim3 threads(tileSize, tileSize);
  dim3 blocksUnpadded((width + tileSize - 1) / tileSize,
                      (height + tileSize - 1) / tileSize);
  dim3 blocksPadded((pitch + tileSize - 1) / tileSize,
                    (height + tileSize - 1) / tileSize);

  int shWidth = tileSize + kSize - 1;
  int shWidthPadded = shWidth + PADDING;
  size_t shMemUnpad = shWidth * shWidth * sizeof(float);
  size_t shMemPadded = shWidthPadded * shWidthPadded * sizeof(float);

  cudaEventRecord(evStart);
  convolveTiled<<<blocksUnpadded, threads, shMemUnpad>>>(input, output_tiled,
                                                         width, height, kSize, tileSize);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&tiledMs, evStart, evStop);

  cudaEventRecord(evStart);
  convolvePadded<<<blocksPadded, threads, shMemPadded>>>(input_padded, output_padded,
                                                         width, height, kSize, tileSize, pitch);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&paddedMs, evStart, evStop);

  bool error = false;
  for (int r = 0; r < height && !error; r++)
    for (int c = 0; c < width && !error; c++)
    {
      if (fabsf(output_cpu[r * width + c] - output_padded[r * pitch + c]) > 1e-3f)
      {
        printf("MISMATCH at [%d][%d]\n", r, c);
        error = true;
      }
    }

  printf("Kernel %dx%d | Tile %dx%d | CPU: %.3f ms | Tiled: %.3f ms | Padded: %.3f ms | Speedup padded vs tiled: %.2fx | %s\n",
         kSize, kSize, tileSize, tileSize, cpuMs, tiledMs, paddedMs,
         tiledMs / paddedMs, error ? "MISMATCH" : "OK");

  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);
  free(kernel2D);
  free(kernel1D);
  cudaFree(input);
  cudaFree(input_padded);
  cudaFree(output_cpu);
  cudaFree(output_tiled);
  cudaFree(output_padded);
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