#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

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

__global__ void convolveGPU(float *input, float *output, float *kernel,
                            int width, int height, int kSize, int boundaryMode)
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

int main()
{
  int width = IMAGE_WIDTH;
  int height = IMAGE_HEIGHT;
  int kSizes[] = {3, 5, 7};

  for (int s = 0; s < 3; s++)
  {
    int kSize = kSizes[s];
    size_t imgSize = (size_t)width * height * sizeof(float);
    size_t krnSize = (size_t)kSize * kSize * sizeof(float);

    float *input, *output_cpu, *output_gpu, *kernel;
    cudaMallocManaged(&input, imgSize);
    cudaMallocManaged(&output_cpu, imgSize);
    cudaMallocManaged(&output_gpu, imgSize);
    cudaMallocManaged(&kernel, krnSize);

    for (int i = 0; i < width * height; i++)
      input[i] = (float)(i % 256) / 255.0f;

    makeGaussianKernel(kernel, kSize, 1.0f);

    cudaMemLocation gpuLoc;
    gpuLoc.type = cudaMemLocationTypeDevice;
    gpuLoc.id = 0;
    cudaMemPrefetchAsync(input, imgSize, gpuLoc, 0);
    cudaMemPrefetchAsync(output_gpu, imgSize, gpuLoc, 0);
    cudaMemPrefetchAsync(kernel, krnSize, gpuLoc, 0);
    cudaDeviceSynchronize();

    clock_t start = clock();
    convolveCPU(input, output_cpu, kernel, width, height, kSize, 0);
    clock_t end = clock();
    double cpuMs = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    float gpuMs = 0;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    cudaEventRecord(evStart);
    convolveGPU<<<numBlocks, threadsPerBlock>>>(input, output_gpu, kernel,
                                                width, height, kSize, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    cudaEventElapsedTime(&gpuMs, evStart, evStop);

    bool error = false;
    for (int i = 0; i < width * height && !error; i++)
    {
      if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-4f)
      {
        printf("MISMATCH at %d: cpu=%.6f gpu=%.6f\n", i, output_cpu[i], output_gpu[i]);
        error = true;
      }
    }

    printf("Kernel %dx%d | CPU: %.3f ms | GPU: %.3f ms | Speedup: %.2fx | %s\n",
           kSize, kSize, cpuMs, gpuMs, cpuMs / gpuMs,
           error ? "MISMATCH" : "OK");

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaFree(input);
    cudaFree(output_cpu);
    cudaFree(output_gpu);
    cudaFree(kernel);
  }

  return 0;
}