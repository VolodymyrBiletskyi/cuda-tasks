#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
    for (int col = 0; col < width; col++)
    {
      float sum = 0.0f;
      for (int kr = -radius; kr <= radius; kr++)
        for (int kc = -radius; kc <= radius; kc++)
        {
          int r = row + kr, c = col + kc;
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
      output[row * width + col] = sum;
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
      for (int kc = -radius; kc <= radius; kc++)
      {
        int r = row + kr, c = col + kc;
        float pixel = 0.0f;
        if (r >= 0 && r < height && c >= 0 && c < width)
          pixel = input[r * width + c];
        sum += pixel * constKernel[(kr + radius) * kSize + (kc + radius)];
      }
    output[row * width + col] = sum;
  }
}

__global__ void convolveTiled(float *input, float *output,
                              int width, int height, int kSize, int tileSize)
{
  extern __shared__ float shInput[];
  int radius = kSize / 2, shWidth = tileSize + kSize - 1;
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = blockIdx.x * tileSize + tx, row = blockIdx.y * tileSize + ty;
  for (int dr = ty; dr < shWidth; dr += tileSize)
    for (int dc = tx; dc < shWidth; dc += tileSize)
    {
      int r = blockIdx.y * tileSize + dr - radius, c = blockIdx.x * tileSize + dc - radius;
      float pixel = 0.0f;
      if (r >= 0 && r < height && c >= 0 && c < width)
        pixel = input[r * width + c];
      shInput[dr * shWidth + dc] = pixel;
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
  int radius = kSize / 2, tx = threadIdx.x;
  int row = blockIdx.y, col = blockIdx.x * blockDim.x + tx;
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
  int radius = kSize / 2, ty = threadIdx.y;
  int col = blockIdx.x, row = blockIdx.y * blockDim.y + ty;
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

__global__ void convolvePadded(float *input, float *output,
                               int width, int height, int kSize,
                               int tileSize, int pitch)
{
  extern __shared__ float shInput[];
  int radius = kSize / 2, shWidth = tileSize + kSize - 1 + PADDING;
  int tx = threadIdx.x, ty = threadIdx.y;
  int col = blockIdx.x * tileSize + tx, row = blockIdx.y * tileSize + ty;
  for (int dr = ty; dr < shWidth; dr += tileSize)
    for (int dc = tx; dc < shWidth - PADDING; dc += tileSize)
    {
      int r = blockIdx.y * tileSize + dr - radius, c = blockIdx.x * tileSize + dc - radius;
      float pixel = 0.0f;
      if (r >= 0 && r < height && c >= 0 && c < width)
        pixel = input[r * pitch + c];
      shInput[dr * shWidth + dc] = pixel;
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
    for (int kc = -radius; kc <= radius; kc++)
    {
      float val = expf(-(kr * kr + kc * kc) / (2.0f * sigma * sigma));
      kernel[(kr + radius) * kSize + (kc + radius)] = val;
      sum += val;
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

void runExperiment(int imgWidth, int imgHeight, int kSize)
{
  int tileSize = 16;
  int pitch = ((imgWidth + 31) / 32) * 32;
  size_t imgSize = (size_t)imgWidth * imgHeight * sizeof(float);
  size_t imgSizePadded = (size_t)pitch * imgHeight * sizeof(float);
  size_t krnSize2D = (size_t)kSize * kSize * sizeof(float);
  size_t krnSize1D = (size_t)kSize * sizeof(float);

  float *input, *input_padded, *output_cpu, *output_naive;
  float *output_tiled, *output_sep, *output_padded, *temp;
  cudaMallocManaged(&input, imgSize);
  cudaMallocManaged(&input_padded, imgSizePadded);
  cudaMallocManaged(&output_cpu, imgSize);
  cudaMallocManaged(&output_naive, imgSize);
  cudaMallocManaged(&output_tiled, imgSize);
  cudaMallocManaged(&output_sep, imgSize);
  cudaMallocManaged(&output_padded, imgSizePadded);
  cudaMallocManaged(&temp, imgSize);

  float *kernel2D = (float *)malloc(krnSize2D);
  float *kernel1D = (float *)malloc(krnSize1D);

  for (int r = 0; r < imgHeight; r++)
    for (int c = 0; c < imgWidth; c++)
    {
      float val = (float)((r * imgWidth + c) % 256) / 255.0f;
      input[r * imgWidth + c] = val;
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
  cudaMemPrefetchAsync(output_naive, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_tiled, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_sep, imgSize, gpuLoc, 0);
  cudaMemPrefetchAsync(output_padded, imgSizePadded, gpuLoc, 0);
  cudaMemPrefetchAsync(temp, imgSize, gpuLoc, 0);
  cudaDeviceSynchronize();

  clock_t start = clock();
  convolveCPU(input, output_cpu, kernel2D, imgWidth, imgHeight, kSize, 0);
  clock_t end = clock();
  double cpuMs = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  float naiveMs = 0, tiledMs = 0, sepMs = 0, paddedMs = 0;

  dim3 t16(16, 16);
  dim3 bNaive((imgWidth + 15) / 16, (imgHeight + 15) / 16);

  cudaEventRecord(evStart);
  convolveNaive<<<bNaive, t16>>>(input, output_naive, imgWidth, imgHeight, kSize);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&naiveMs, evStart, evStop);

  dim3 bTiled((imgWidth + tileSize - 1) / tileSize, (imgHeight + tileSize - 1) / tileSize);
  int shW = tileSize + kSize - 1;
  cudaEventRecord(evStart);
  convolveTiled<<<bTiled, t16, shW * shW * sizeof(float)>>>(input, output_tiled, imgWidth, imgHeight, kSize, tileSize);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&tiledMs, evStart, evStop);

  dim3 hT(128, 1);
  dim3 hB((imgWidth + 127) / 128, imgHeight);
  dim3 vT(1, 128);
  dim3 vB(imgWidth, (imgHeight + 127) / 128);
  cudaEventRecord(evStart);
  convolveHorizontal<<<hB, hT, (128 + kSize - 1) * sizeof(float)>>>(input, temp, imgWidth, imgHeight, kSize);
  cudaDeviceSynchronize();
  convolveVertical<<<vB, vT, (128 + kSize - 1) * sizeof(float)>>>(temp, output_sep, imgWidth, imgHeight, kSize);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&sepMs, evStart, evStop);

  dim3 bPadded((pitch + tileSize - 1) / tileSize, (imgHeight + tileSize - 1) / tileSize);
  int shWP = tileSize + kSize - 1 + PADDING;
  cudaEventRecord(evStart);
  convolvePadded<<<bPadded, t16, shWP * shWP * sizeof(float)>>>(input_padded, output_padded, imgWidth, imgHeight, kSize, tileSize, pitch);
  cudaDeviceSynchronize();
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&paddedMs, evStart, evStop);

  printf("Img %dx%d | Kernel %dx%d | CPU: %7.2f ms | Naive: %6.3f ms | Tiled: %6.3f ms | Sep: %6.3f ms | Padded: %6.3f ms\n",
         imgWidth, imgHeight, kSize, kSize, cpuMs, naiveMs, tiledMs, sepMs, paddedMs);

  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);
  free(kernel2D);
  free(kernel1D);
  cudaFree(input);
  cudaFree(input_padded);
  cudaFree(output_cpu);
  cudaFree(output_naive);
  cudaFree(output_tiled);
  cudaFree(output_sep);
  cudaFree(output_padded);
  cudaFree(temp);
}

int main()
{
  int imgSizes[] = {256, 512, 1024, 2048};
  int kSizes[] = {3, 5, 7};

  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 3; k++)
      runExperiment(imgSizes[i], imgSizes[i], kSizes[k]);

  return 0;
}