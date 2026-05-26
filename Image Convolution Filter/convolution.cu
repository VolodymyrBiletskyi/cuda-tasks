#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define KERNEL_SIZE 5

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
  int numSizes = 3;

  for (int s = 0; s < numSizes; s++)
  {
    int kSize = kSizes[s];
    size_t imgSize = (size_t)width * height * sizeof(float);
    size_t krnSize = (size_t)kSize * kSize * sizeof(float);

    float *input = (float *)malloc(imgSize);
    float *output = (float *)malloc(imgSize);
    float *kernel = (float *)malloc(krnSize);

    for (int i = 0; i < width * height; i++)
      input[i] = (float)(i % 256) / 255.0f;

    makeGaussianKernel(kernel, kSize, 1.0f);

    clock_t start = clock();
    convolveCPU(input, output, kernel, width, height, kSize, 0);
    clock_t end = clock();
    double msZero = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    start = clock();
    convolveCPU(input, output, kernel, width, height, kSize, 1);
    end = clock();
    double msClamp = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("Kernel %dx%d | zero padding: %.3f ms | clamp: %.3f ms\n",
           kSize, kSize, msZero, msClamp);

    free(input);
    free(output);
    free(kernel);
  }

  return 0;
}