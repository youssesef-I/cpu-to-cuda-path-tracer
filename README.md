# Benchmark Methodology

This directory documents the performance measurements for the CPU and CUDA implementations of the path tracer.

The summarized measurements are stored in [`results.csv`](results.csv).

## Hardware

* **CPU:** Intel Core i7-13700K
* **GPU:** NVIDIA GeForce RTX 4090
* **GPU architecture:** Ada Lovelace
* **CUDA compute capability target:** `sm_89`

## Render Configuration

All reported implementations rendered the same scene using:

* **Resolution:** 1200×675
* **Samples per pixel:** 500
* **Maximum ray bounces:** 10
* **Scene:** One diffuse sphere above a large sphere used as the ground plane
* **Lighting:** Procedural sky illumination

The implementations use the same scene and rendering configuration, but they do not use identical numerical precision.

## CPU Baseline

The CPU implementation was compiled with:

```sh
clang++ -O2 Raytrace.cpp -o rt
```

The current CPU implementation:

* runs on one CPU thread
* uses double-precision floating-point arithmetic
* evaluates pixels through a serial loop

Measured render time:

```text
22.9 seconds
```

This measurement is the baseline used for the reported speedup values.

## CUDA Build

The CUDA implementation was compiled with:

```sh
nvcc -O2 -ccbin g++-14 -arch=sm_89 Raytracer.cu -o rtgpu
```

The CUDA implementation:

* uses single-precision floating-point arithmetic
* assigns one GPU thread to each pixel
* stores the scene as contiguous sphere data
* replaces recursive ray evaluation with an iterative bounce loop
* maintains independent cuRAND state for each pixel

## CUDA Timing

GPU kernel execution time was measured using CUDA events.

The reported GPU values measure the rendering kernel and exclude:

* initial CUDA context creation
* application startup
* output-file encoding
* one-time host/device setup costs

This isolates rendering throughput, but it is not an end-to-end application latency measurement.

### Initial CUDA Port

The initial CUDA implementation used rejection sampling when generating random unit vectors.

Measured kernel time:

```text
~18.5 milliseconds
```

Relative to the single-threaded CPU implementation:

```text
~1238× speedup
```

### Analytic Sphere Sampling

The optimized implementation replaces the rejection-sampling loop with a fixed-work analytic construction using a uniformly sampled vertical coordinate and azimuth.

Measured kernel time:

```text
~15.2 milliseconds
```

Relative to the single-threaded CPU implementation:

```text
~1507× speedup
```

Relative to the initial CUDA port, this represents an approximately:

```text
17.8% reduction in kernel execution time
```

The main README rounds this result to approximately **18%**.

## Interpretation

These measurements compare the current implementations, not the theoretical maximum performance of the CPU and GPU.

Important limitations include:

1. The CPU baseline is single-threaded.
2. The CPU implementation uses `double`, while the CUDA implementation uses `float`.
3. GPU measurements are kernel-only rather than end-to-end wall-clock measurements.
4. The scene contains only two spheres and does not use an acceleration structure.

A future multithreaded CPU implementation would provide a stronger CPU baseline. Additional CUDA profiling with Nsight Compute could also identify the next optimization target.

## Reproducing the Builds

From the repository root, build both implementations with:

```sh
make
```

Run the CPU implementation with:

```sh
make run-cpu
```

Run the CUDA implementation with:

```sh
make run-gpu
```

Remove generated binaries and PPM images with:

```sh
make clean
```

