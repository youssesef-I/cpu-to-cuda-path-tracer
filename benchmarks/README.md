# Benchmark Methodology

This directory records the performance measurements used in the main project
README.

## Test Configuration

- CPU: Intel Core i7-13700K
- GPU: NVIDIA RTX 4090
- CUDA toolkit: 12.9
- Nsight Compute: 2025.2
- CUDA architecture target: `sm_89`
- Resolution: 1200 x 675
- Samples per pixel: 500
- Maximum ray bounces: 10

CPU build:

```sh
clang++ -O2 Raytrace.cpp -o rt
```

Precise GPU build:

```sh
nvcc -O2 -arch=sm_89 -ccbin g++-14 Raytracer.cu -o rtgpu-precise
```

Final fast-math GPU build:

```sh
nvcc -O2 --use_fast_math -arch=sm_89 -ccbin g++-14 Raytracer.cu -o rtgpu
```

## Performance Progression

| Variant | Time | Notes |
| --- | ---: | --- |
| Single-threaded CPU | 22.9 s | CPU reference baseline |
| Initial CUDA port | ~18.5 ms | Initial GPU implementation |
| Analytic sphere sampling | 15.04 ms | 10-run average |
| Fast math | 10.73 ms | 10-run average, `--use_fast_math` |

Replacing rejection-based random sphere sampling with analytic uniform-sphere
sampling reduced kernel time by approximately 18.7% relative to the initial
CUDA implementation.

After profiling the optimized kernel with NVIDIA Nsight Compute, enabling
fast-math reduced average kernel time by a further 28.7%.

From the initial CUDA implementation to the final fast-math build, kernel time
fell by approximately 42%.

## Ten-Run Timing

Precise optimized build:

```text
14.6053
15.0804
14.8613
15.2534
15.3334
14.9009
15.4358
14.6225
14.5961
15.7153
```

Mean:

```text
15.04 ms
```

Fast-math build:

```text
11.0394
10.6803
11.1657
10.2437
10.7459
10.8420
10.0536
10.9711
11.0994
10.4479
```

Mean:

```text
10.73 ms
```

## Nsight Compute Findings

The precise optimized kernel was compute-heavy rather than
DRAM-bandwidth-bound.

Selected precise-build measurements:

| Metric | Value |
| --- | ---: |
| Nsight kernel duration | 15.88 ms |
| Registers per thread | 66 |
| Theoretical occupancy | 58.33% |
| Achieved occupancy | 50.29% |
| Average active threads per warp | 14.38 |
| Executed instructions | 14.05 billion |
| Branch efficiency | 94.79% |
| L1/TEX hit rate | 99.93% |

Selected fast-math measurements:

| Metric | Value |
| --- | ---: |
| Nsight kernel duration | 10.98 ms |
| Registers per thread | 59 |
| Theoretical occupancy | 66.67% |
| Achieved occupancy | 56.08% |
| Average active threads per warp | 16.01 |
| Executed instructions | 9.26 billion |
| Branch efficiency | 92.37% |
| L1/TEX hit rate | 99.93% |

Fast math reduced the measured dynamic executed-instruction count from roughly
14.05 billion to 9.26 billion, a reduction of approximately 34%.

It also reduced register demand and increased both theoretical and achieved
occupancy.

## Output Comparison

The CUDA implementation initializes cuRAND deterministically for each pixel:

```cpp
curand_init(1984 + pixel_index, 0, 0, &state);
```

Because `--use_fast_math` changes floating-point semantics, its output is not
expected to be bit-identical to the precise build.

ImageMagick RMSE comparison between the two deterministic renders produced:

```text
28.0014 (0.000427275)
```

The normalized RMSE was therefore:

```text
0.000427275
```

The images were additionally inspected side-by-side for visible differences.

## Timing Scope

The CPU value measures the single-threaded CPU renderer while the CUDA values
measure GPU kernel execution.

The quoted CPU/GPU speedups are therefore useful project-level comparisons,
not claims that the CPU and GPU numbers represent perfectly identical
end-to-end timing scopes.
