# CUDA Image Convolution Filter

## Test Environment

| Component     | Spec                                            |
| ------------- | ----------------------------------------------- |
| CPU           | AMD Ryzen 7 260, 8 cores / 16 threads, 3.80 GHz |
| GPU           | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM   |
| RAM           | 16 GB DDR5 5600 MHz (Micron)                    |
| NVIDIA Driver | 610.47 (NVIDIA-SMI 610.47, KMD 610.47)          |
| CUDA Version  | 13.3                                            |
| OS            | Windows 10 Pro                                  |

Six implementations are compared: CPU, naive GPU, tiled shared memory, constant memory, separable, and padded is applied to float images up to 2048×2048 with kernel sizes 3×3, 5×5, and 7×7. Each version targets a specific bottleneck, from uncoalesced global reads to bank conflicts. All results are timed with CUDA events and verified against the CPU reference.

## Prerequisites

- CUDA Toolkit (nvcc)
- NVIDIA GPU with CUDA support

## Build & Run

```bash
nvcc -o convolution convolution.cu
convolution
```

Output shows CPU time, GPU time, and speedup for each kernel size and tile configuration.

## How It Works

```c
#define MAX_KERNEL_SIZE 49
#define MAX_KERNEL_1D   7
#define PADDING         1

__constant__ float constKernel[MAX_KERNEL_SIZE];
__constant__ float constKernel1D[MAX_KERNEL_1D];
```

`MAX_KERNEL_SIZE` covers the largest 2D kernel (7×7 = 49 elements). Two separate constant memory arrays are declared one for the full 2D kernel used by naive, tiled, and padded kernels, and one 1D kernel used by the separable passes. `PADDING` is the +1 shared memory padding that eliminates bank conflicts.

```c
void makeGaussianKernel2D(float *kernel, int kSize, float sigma) {
    int radius = kSize / 2;
    float sum = 0.0f;
    for (int kr = -radius; kr <= radius; kr++)
        for (int kc = -radius; kc <= radius; kc++) {
            float val = expf(-(kr*kr + kc*kc) / (2.0f * sigma * sigma));
            kernel[(kr+radius)*kSize + (kc+radius)] = val;
            sum += val;
        }
    for (int i = 0; i < kSize*kSize; i++)
        kernel[i] /= sum;
}
```

Fills a kSize×kSize array with Gaussian values using the formula `e^(-(x²+y²) / 2σ²)`, then normalizes by dividing every element by the total sum so the kernel weights add up to 1. Without normalization the output image would get brighter or darker. `sigma = 1.0` is used throughout controls how wide the blur spreads.

```c
void convolveCPU(float *input, float *output, float *kernel,
                 int width, int height, int kSize, int boundaryMode) {
    int radius = kSize / 2;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int kr = -radius; kr <= radius; kr++)
                for (int kc = -radius; kc <= radius; kc++) {
                    int r = row + kr, c = col + kc;
                    float pixel = 0.0f;
                    if (boundaryMode == 0) {
                        if (r >= 0 && r < height && c >= 0 && c < width)
                            pixel = input[r * width + c];
                    } else {
                        r = r < 0 ? 0 : (r >= height ? height-1 : r);
                        c = c < 0 ? 0 : (c >= width  ? width-1  : c);
                        pixel = input[r * width + c];
                    }
                    sum += pixel * kernel[(kr+radius)*kSize + (kc+radius)];
                }
            output[row * width + col] = sum;
        }
}
```

Four nested loops image rows, image columns, kernel rows, kernel columns. For each output pixel it accumulates the weighted sum of the neighborhood. `boundaryMode 0` zero-pads out-of-bounds pixels; `boundaryMode 1` clamps coordinates to the nearest valid edge pixel. Used as the correctness reference for all GPU versions.

```c
__global__ void convolveNaive(float *input, float *output,
                               int width, int height, int kSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int radius = kSize / 2;
        float sum = 0.0f;
        for (int kr = -radius; kr <= radius; kr++)
            for (int kc = -radius; kc <= radius; kc++) {
                int r = row + kr, c = col + kc;
                float pixel = 0.0f;
                if (r >= 0 && r < height && c >= 0 && c < width)
                    pixel = input[r * width + c];
                sum += pixel * constKernel[(kr+radius)*kSize + (kc+radius)];
            }
        output[row * width + col] = sum;
    }
}
```

One thread per pixel. Each thread runs the full k² inner loop, reading input pixels from global memory and kernel weights from constant memory. The `if` guard handles image boundaries with zero padding. Global memory reads are the bottleneck, every thread independently fetches its neighborhood with no data reuse between threads.

```c
__global__ void convolveTiled(float *input, float *output,
                               int width, int height, int kSize, int tileSize) {
    extern __shared__ float shInput[];
    int radius = kSize / 2, shWidth = tileSize + kSize - 1;
    // load halo region into shInput
    __syncthreads();
    // compute from shared memory only
    for (int kr = 0; kr < kSize; kr++)
        for (int kc = 0; kc < kSize; kc++)
            sum += shInput[(ty + kr) * shWidth + (tx + kc)] * constKernel[kr * kSize + kc];
}
```

Each block loads a halo region of size `(tileSize + kSize - 1)²` into shared memory.The output tile plus `radius` pixels of border on each side needed for the convolution. After `__syncthreads()` all k² reads per pixel come from on-chip shared memory instead of global memory. `shWidth = tileSize + kSize - 1` is the shared memory row width including the halo.

```c
__constant__ float constKernel[MAX_KERNEL_SIZE];
cudaMemcpyToSymbol(constKernel,  kernel2D, krnSize2D);
cudaMemcpyToSymbol(constKernel1D, kernel1D, krnSize1D);
```

Kernel weights are copied to constant memory before any kernel launch. When all 32 threads in a warp read the same kernel index simultaneously the hardware broadcasts a single value to all of them, one memory transaction instead of 32. Two symbols are used: `constKernel` for 2D kernels, `constKernel1D` for the separable 1D passes.

```c
convolveHorizontal<<<hB, hT, (128 + kSize - 1) * sizeof(float)>>>(input, temp, width, height, kSize);
cudaDeviceSynchronize();
convolveVertical<<<vB, vT, (128 + kSize - 1) * sizeof(float)>>>(temp, output, width, height, kSize);
```

Horizontal pass uses 128×1 thread blocks, one row of 128 threads per block and reads each row into shared memory with its left and right halo, then multiplies by `constKernel1D`. The result goes into a `temp` buffer. Vertical pass uses 1×128 thread blocks, reads a column segment into shared memory, and applies `constKernel1D` again. Together they reduce per-pixel operations from k² to 2k.

```c
int pitch = ((imgWidth + 31) / 32) * 32;
```

Rounds the row width up to the nearest multiple of 32 floats (128 bytes). This ensures every row starts at a 128-byte aligned address exactly one GPU cache line. When 32 threads in a warp read 32 consecutive floats from an aligned address the hardware services it in a single memory transaction. The `+1` padding (`PADDING 1`) on shared memory rows prevents bank conflicts when threads access along columns.

```c
cudaMemPrefetchAsync(input, imgSize, gpuLoc, 0);
cudaMemPrefetchAsync(input_padded, imgSizePadded, gpuLoc, 0);
cudaDeviceSynchronize();
```

Since all allocations use `cudaMallocManaged`, data lives in unified memory and migrates on demand. Without prefetching the first kernel access would trigger page faults, stalling execution. These calls move all buffers to the GPU upfront so kernels start immediately without migration overhead.

```c
cudaEventRecord(evStart);
convolveNaive<<<bNaive, t16>>>(input, output_naive, imgWidth, imgHeight, kSize);
cudaDeviceSynchronize();
cudaEventRecord(evStop);
cudaEventElapsedTime(&naiveMs, evStart, evStop);
```

CUDA events are hardware timestamps placed on the GPU timeline. `cudaEventRecord` drops a marker, `cudaDeviceSynchronize` waits for the kernel to complete, then `cudaEventElapsedTime` returns the exact elapsed time in milliseconds. This is more precise than CPU `clock()` and measures only GPU execution time, not host overhead.

## Performance Results

### Full benchmark (image sizes 256→2048, tile 16×16)

| Image     | Kernel | CPU       | Naive     | Tiled    | Separable | Padded   |
| --------- | ------ | --------- | --------- | -------- | --------- | -------- |
| 256×256   | 3×3    | 4.00 ms   | 1.008 ms  | 0.182 ms | 0.256 ms  | 0.055 ms |
| 256×256   | 7×7    | 12.00 ms  | 2.884 ms  | 0.047 ms | 0.055 ms  | 0.058 ms |
| 1024×1024 | 3×3    | 60.00 ms  | 8.475 ms  | 0.140 ms | 0.194 ms  | 0.110 ms |
| 1024×1024 | 7×7    | 197.00 ms | 10.188 ms | 0.212 ms | 0.262 ms  | 0.181 ms |
| 2048×2048 | 3×3    | 238.00 ms | 33.721 ms | 0.406 ms | 0.664 ms  | 0.396 ms |
| 2048×2048 | 7×7    | 732.00 ms | 39.045 ms | 0.698 ms | 0.928 ms  | 0.657 ms |

At 2048×2048 with 7×7 kernel: padded GPU finishes in **0.66 ms vs 732 ms on CPU**.

## Conclusions

The naive GPU gave only 12× speedup at 1024×1024 due to uncoalesced global memory reads, the bottleneck was bandwidth, not compute. Shared memory tiling pushed that to 683×, and padded rows eliminated remaining bank conflicts to reach 722×. Separable convolution reduced per-pixel operations from k² to 2k but the overhead of two kernel launches landed it at 438×. All GPU versions scaled linearly with image size while CPU cost grew with both image size and kernel size, confirming that the optimized kernels shift the bottleneck from memory to compute exactly where the GPU excels.
