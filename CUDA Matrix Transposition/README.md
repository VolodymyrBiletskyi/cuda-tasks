# CUDA Matrix Transposition

## Test Environment

| Component     | Spec                                            |
| ------------- | ----------------------------------------------- |
| CPU           | AMD Ryzen 7 260, 8 cores / 16 threads, 3.80 GHz |
| GPU           | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM   |
| RAM           | 16 GB DDR5 5600 MHz (Micron)                    |
| NVIDIA Driver | 610.47 (NVIDIA-SMI 610.47, KMD 610.47)          |
| CUDA Version  | 13.3                                            |
| OS            | Windows 10 Pro                                  |

Three implementations are compared - CPU, naive GPU, and shared memory optimized GPU on an N×N float matrix. The naive GPU suffers from uncoalesced writes despite parallelism; the optimized version stages data through shared memory to make both reads and writes coalesced. All results are timed with CUDA events and verified against the CPU reference.

## Prerequisites

- CUDA Toolkit (nvcc)
- NVIDIA GPU with CUDA support

## Build & Run

```bash
nvcc -o transpose transpose.cu
transpose
```

Output shows timing for each implementation and a correctness check against the CPU reference.

## How It Works

```c
#define N 1024
#define TILE_WIDTH 32
```

`N` is the matrix size. `TILE_WIDTH` controls both the thread block dimensions and the shared memory tile size.It's a compile-time constant because shared memory arrays must have a fixed size known at compile time.

```c
void transposeCPU(float *in, float *out, int n) {
    for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++)
            out[col * n + row] = in[row * n + col];
}
```

Reads element at `[row][col]` and writes it to `[col][row]`. Simple reference implementation used to verify that GPU results are correct.

```c
__global__ void transposeNaive(float *in, float *out, int n) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n && col < n)
        out[col * n + row] = in[row * n + col];
}
```

Each thread computes its own `row` and `col` from its block and thread indices. The `if` guard prevents out-of-bounds access when the matrix size isn't a multiple of the tile. The write `out[col * n + row]` is the problematic line.This is the uncoalesced scatter.

```c
__global__ void transposeOptimized(float *in, float *out, int n) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    if (row < n && col < n)
        tile[threadIdx.y][threadIdx.x] = in[row * n + col]; // coalesced read

    __syncthreads();

    col = blockIdx.y * TILE_WIDTH + threadIdx.x;
    row = blockIdx.x * TILE_WIDTH + threadIdx.y;

    if (row < n && col < n)
        out[row * n + col] = tile[threadIdx.x][threadIdx.y]; // coalesced write
}
```

Phase 1 - threads read a tile from global memory row by row into shared memory. Consecutive threads have consecutive `col` values, so reads are coalesced. `__syncthreads()` makes sure every thread has finished loading before anyone reads from the tile. Phase 2 - block indices are swapped (`blockIdx.x` ↔ `blockIdx.y`), so now threads write to the transposed position. Consecutive threads again have consecutive addresses in global memory. The `+1` on the tile width means each row is offset by one slot, preventing multiple threads from hitting the same shared memory bank.

```c
cudaMemPrefetchAsync(in, size, gpuLoc, 0);
```

Since the code uses unified memory (`cudaMallocManaged`), data can live on either CPU or GPU. Without prefetching, the GPU would page-fault on first access and fetch data on demand, slow. This call tells the driver to move the data to the GPU upfront before the kernels run.

```c
cudaEventRecord(evStart);
transposeNaive<<<blocks, threads>>>(in, out_naive, n);
cudaDeviceSynchronize();
cudaEventRecord(evStop);
cudaEventElapsedTime(&milliseconds, evStart, evStop);
```

CUDA events are hardware timestamps on the GPU timeline. `cudaEventRecord` drops a timestamp, `cudaDeviceSynchronize` waits for the kernel to finish, then `cudaEventElapsedTime` gives the exact time between the two stamps in milliseconds, much more precise than CPU `clock()`.

## Performance Results

### Block Size Sweep (1024×1024)

| Kernel    | Block 8×8 | Block 16×16  | Block 32×32 |
| --------- | --------- | ------------ | ----------- |
| Naive GPU | 20.525 ms | 0.045 ms     | 0.119 ms    |
| Optimized | 0.119 ms  | **0.064 ms** | 0.050 ms    |

### Scaling Across Matrix Sizes (16×16 blocks)

| Matrix Size | CPU       | Naive GPU | Optimized GPU | Speedup vs CPU |
| ----------- | --------- | --------- | ------------- | -------------- |
| 512×512     | 1.000 ms  | 0.985 ms  | 0.067 ms      | **14.82×**     |
| 1024×1024   | 5.000 ms  | 2.385 ms  | 0.064 ms      | **78.12×**     |
| 2048×2048   | 23.000 ms | 10.418 ms | 0.142 ms      | **161.52×**    |

### Scaling Across Matrix Sizes (32×32 blocks)

| Matrix Size | CPU       | Naive GPU   | Optimized GPU | Speedup vs CPU | Speedup vs Naive |
| ----------- | --------- | ----------- | ------------- | -------------- | ---------------- |
| 512×512     | —         | 93.334 ms\* | 0.098 ms      | —              | 947.60×          |
| 1024×1024   | 4.000 ms  | 2.418 ms    | 0.050 ms      | **79.57×**     | 48.09×           |
| 2048×2048   | 18.000 ms | 10.791 ms   | 0.180 ms      | **99.96×**     | 59.93×           |

\*512×512 CPU time rounds to 0 ms due to Windows `clock()` resolution (~1 ms). Naive GPU at 512×512 shows GPU driver warmup on first launch, not actual kernel speed.

## Conclusions

Naive GPU barely outperformed CPU because uncoalesced writes serialized memory transactions, eliminating most of the parallelism benefit. Shared memory tiling fixed that by staging data through on-chip memory, achieving up to 161× speedup over CPU at 2048×2048. CPU time scaled 23× from 512 to 2048 while the optimized GPU scaled only 2×, confirming that GPU efficiency grows with problem size. Block size had a measurable impact, 32×32 won at medium sizes but lost at 2048×2048 due to reduced occupancy at the thread limit.
