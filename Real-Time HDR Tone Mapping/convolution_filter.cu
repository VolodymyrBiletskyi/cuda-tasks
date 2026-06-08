#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Example filter kernels students can implement:

// Box blur (3x3)
float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

// Gaussian blur (5x5)
float gaussianBlur5x5[25] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f};

// Sobel edge detection (horizontal)
float sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1};

// Sobel edge detection (vertical)
float sobelY[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1};

// Sharpen filter
float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0};

// Utility to check for CUDA errors
#define CHECK_CUDA_ERROR(call)                                        \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess)                                       \
        {                                                             \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                    cudaGetErrorString(err), __LINE__, __FILE__);     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    }

// Structure to hold image data
typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels; // 1 for grayscale, 3 for RGB, 4 for RGBA
} Image;

// CPU implementation of 2D convolution
void convolutionCPU(const Image *input, Image *output, const float *filter, int filterWidth)
{
    // TODO: Implement CPU version of convolution
}

// Naive GPU implementation - each thread computes one output pixel
__global__ void convolutionKernelNaive(unsigned char *input, unsigned char *output,
                                       float *filter, int filterWidth,
                                       int width, int height, int channels)
{
    // TODO: Implement naive GPU version of convolution
}

// Shared memory implementation
__global__ void convolutionKernelShared(unsigned char *input, unsigned char *output,
                                        float *filter, int filterWidth,
                                        int width, int height, int channels)
{
    // TODO: Implement shared memory version of convolution
}

// Constants for filter definitions
__constant__ float d_filter[81]; // Max 9x9 filter

// Main function to compare implementations
int main(int argc, char **argv)
{
    // TODO: Load or generate an image

    // TODO: Define convolution filters (e.g., blur, sharpen, edge detection)

    // TODO: Implement timing utilities

    // TODO: Run CPU implementation

    // TODO: Run GPU implementations

    // TODO: Compare results and performance

    return 0;
}