#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define MAX_KERNEL_SIZE 49
#define MAX_KERNEL_1D 7

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

__global__ void convolveHorizontal(float *input, float *output,
                                   int width, int height, int kSize)
{
  extern __shared__ float shRow[];

  int radius = kSize / 2;
  int tx = threadIdx.x;
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + tx;
  int shWidth = blockDim.x + kSize - 1;

  for (int dc = tx; dc < shWidth; dc += blockDim.x)
  {
    int c = blockIdx.x * blockDim.x + dc - radius;
    float pixel = 0.0f;
    if (c >= 0 && c < width && row < height)
      pixel = input[row * width + c];
    shRow[dc] = pixel;
  }

  __syncthreads();

  if (row < height && col < width)
  {
    float sum = 0.0f;
    for (int kc = 0; kc < kSize; kc++)
      sum += shRow[tx + kc] * constKernel1D[kc];
    output[row * width + col] = sum;
  }
}

__global__ void convolveVertical(float *input, float *output,
                                 int width, int height, int kSize)
{
  extern __shared__ float shCol[];

  int radius = kSize / 2;
  int ty = threadIdx.y;
  int col = blockIdx.x;
  int row = blockIdx.y * blockDim.y + ty;
  int shHeight = blockDim.y + kSize - 1;

  for (int dr = ty; dr < shHeight; dr += blockDim.y)
  {
    int r = blockIdx.y * blockDim.y + dr - radius;
    float pixel = 0.0f;
    if (r >= 0 && r < height && col < width)
      pixel = input[r * width + col];
    shCol[dr] = pixel;
  }

  __syncthreads();

  if (row < height && col < width)
  {
    float sum = 0.0f;
    for (int kr = 0; kr < kSize; kr++)
      sum += shCol[ty + kr] * constKernel1D[kr];
    output[row * width + col] = sum;
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
  size_t imgSize = (size_t)width * height * sizeof(float);
  size_t krnSize2D = (size_t)kSize * kSize * sizeof(float);
  size_t krnSize1D = (size_t)kSize * sizeof(float);

  float *input, *output_cpu, *output_tiled, *output_sep, *temp;
  cudaMallocManaged(&input, imgSize);
  cudaMallocManaged(&output_cpu, imgSize);
  cudaMallocManaged(&output_tiled, imgSize);
  cudaMallocManaged(&output_sep, imgSize);
  cudaMallocManaged(&temp, imgSize);

  float *kernel2D = (float *)malloc(krnSize2D);
  float *kernel1D = (float *)malloc(krnSize1D);

  for (int i = 0; i < width * height; i++)
    input[i] = (float)(i % 256) / 255.0f;

  makeGaussianKernel2D(kernel2D, kSize, 1.0f);
  makeGaussianKernel1D(kernel1D, kSize, 1.0f);

  cudaMemcpyToSymbol(constKernel, kernel2D, krnSize2D);
  cudaMemcpyToSymbol(constKernel1D, kernel1D, krnSize1D);

  cudaMemLocation gpuLoc;
  gpuLoc.type = cudaMemLocationTypeDevice;
  gpuLoc.id = 0;
  cudaMemPrefetchAsync(input, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_tiled, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_sep, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(temp, imgSize, gpuLoc, 0);
  cudaDeviceSynchronize();

  clock_t start = clock();
  convolveCPU(input, output_cpu, kernel2D, width, height, kSize, 0);
  clock_t end = clock();
  double cpuMs = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  float tiledMs = 0, sepMs = 0;

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

  dim3 hThreads(128, 1);
  dim3 hBlocks((width + 127) / 128, height);
  size_t hShMem = (128 + kSize - 1) * sizeof(float);

  dim3 vThreads(1, 128);
  dim3 vBlocks(width, (height + 127) / 128);
  size_t vShMem = (128 + kSize - 1) * sizeof(float);

  cudaEventRecord(evStart);
  convolveHorizontal<<<hBlocks, hThreads, hShMem>>>(input, temp, width, height, kSize);
  cudaDeviceSynchronize();
  convolveVertical<<<vBlocks, vThreads, vShMem>>>(temp, output_sep, width, height, kSize);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&sepMs, evStart, evStop);

  bool error = false;
  for (int i = 0; i < width * height && !error; i++)
  {
    if (fabsf(output_cpu[i] - output_sep[i]) > 1e-3f)
    {
      printf("MISMATCH at %d: cpu=%.6f sep=%.6f\n", i, output_cpu[i], output_sep[i]);
      error = true;
    }
  }

  printf("Kernel %dx%d | Tile %dx%d | CPU: %.3f ms | Tiled 2D: %.3f ms | Separable: %.3f ms | Speedup sep vs tiled: %.2fx | %s\n",
         kSize, kSize, tileSize, tileSize, cpuMs, tiledMs, sepMs,
         tiledMs / sepMs, error ? "MISMATCH" : "OK");

  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);
  free(kernel2D);
  free(kernel1D);
  cudaFree(input);
  cudaFree(output_cpu);
  cudaFree(output_tiled);
  cudaFree(output_sep);
  cudaFree(temp);
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