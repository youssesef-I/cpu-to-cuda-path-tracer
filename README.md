# CPU-to-CUDA PATH TRACER
# CPU-to-CUDA Path Tracer

A path tracer implemented first in C++ on the CPU, then ported to CUDA and optimized for GPU execution.

The renderer traces diffuse light transport through a scene of spheres under procedural sky illumination, with antialiasing, multiple ray bounces, and gamma correction. The CUDA port restructures the original CPU implementation around the GPU execution model: the serial pixel loop becomes **one GPU thread per pixel**, recursive ray evaluation becomes iterative, polymorphic scene objects become contiguous concrete data, and per-thread random sampling is redesigned to reduce warp divergence.

At matched scene settings, the CUDA kernel renders in **~15.2 ms**, compared with **~22.9 s** for the original single-threaded CPU implementation — approximately a **1500× measured speedup between these implementations**. An additional optimization targeting RNG-induced warp divergence improved the CUDA implementation by approximately **18%**.

---

## Results

Both implementations render the same scene with the same resolution, samples per pixel, and maximum ray depth.

| Implementation                                 | Render time | Speedup vs. CPU baseline |
| ---------------------------------------------- | ----------: | -----------------------: |
| CPU — single-threaded, `-O2`, double precision |      22.9 s |                       1× |
| GPU — initial CUDA port, single precision      |    ~18.5 ms |                   ~1240× |
| GPU — optimized RNG sampling                   |    ~15.2 ms |                   ~1510× |

**Benchmark system:** RTX 4090 (Ada, `sm_89`) and Intel i7-13700K
**Configuration:** 1200×675, 500 samples/pixel, 10 maximum bounces

GPU timings are kernel execution time measured using CUDA events and reported from warm runs. Run-to-run variation was approximately 4%, so the figures should be treated as approximate rather than exact.

### Benchmark interpretation

The ~1500× result is the measured difference between the current CPU and GPU implementations under these benchmark conditions, rather than a claim that an RTX 4090 is inherently 1500× faster than an i7-13700K.

There are several important qualifications.

**The CPU implementation is single-threaded.**
The original renderer executes its pixel loop serially. A multithreaded implementation using all CPU cores would provide a substantially stronger CPU baseline. Implementing and measuring that version is future work; until then, the 1500× figure should specifically be interpreted as GPU versus the original single-threaded implementation.

**GPU timing is kernel-only.**
CUDA events measure the rendering kernel itself and exclude one-time CUDA context initialization and host/device setup costs. This isolates rendering throughput, but it is different from measuring total application startup-to-image latency. An end-to-end benchmark would therefore be useful as a separate measurement.

**Precision differs.**
The CPU implementation uses `double`, while the CUDA implementation uses `float`. Single precision is conventional for this rendering workload and maps much better to consumer GPU hardware, but it means the implementations are not numerically bit-identical. The benchmark therefore represents the performance of the two practical implementations rather than a controlled precision-for-precision hardware comparison.

---

## The Scene

The scene is deliberately minimal: a small diffuse sphere above a much larger sphere acting as the ground plane, illuminated only by a procedural blue-to-white sky.

Spheres provide a useful primitive for studying the renderer because ray-sphere intersection reduces to a quadratic while still exercising the important parts of a path tracer:

* stochastic ray sampling
* multiple light bounces
* diffuse scattering
* accumulated throughput
* sky illumination
* antialiasing

The project deliberately keeps scene complexity low because its primary focus is the **CPU-to-GPU transformation and optimization**, rather than implementing a complete production renderer.

---

## From CPU to GPU

The CUDA implementation required more than adding CUDA annotations to the original program. Several parts of the CPU design were restructured around the GPU's execution model.

### One thread per pixel

Ray tracing exposes large amounts of data parallelism because each image pixel can be evaluated independently.

The CPU implementation performs rendering through nested serial loops over image coordinates. In the CUDA implementation, those loops are replaced by a kernel launch in which each GPU thread calculates its pixel coordinate from `blockIdx`, `blockDim`, and `threadIdx`.

Conceptually:

```text
CPU:
for every row
    for every pixel
        trace pixel

GPU:
launch thousands of threads
each thread traces one pixel
```

Each thread performs all samples and ray bounces associated with its pixel before writing the final value to the output image.

The serial image traversal is therefore replaced by the kernel launch geometry itself.

---

### Host/device-compatible math types

Types needed by the renderer inside the CUDA kernel — including vectors, rays, and spheres — are made callable from device code using CUDA's `__host__` and `__device__` annotations where required.

The GPU implementation also uses `float` rather than the CPU renderer's `double`, reducing arithmetic and storage cost while using the precision normally required for this workload.

---

### Removing runtime polymorphism

The CPU implementation originally used an abstract `hittable` interface, virtual `hit()` functions, and dynamically allocated objects managed through `shared_ptr`.

That design is convenient for extending a CPU renderer with many primitive types, but it is less attractive for this small GPU workload. Indirect virtual calls can inhibit inlining and introduce additional control-flow and call overhead, while pointer-based object layouts also provide worse locality than contiguous concrete data.

Because this scene contains only spheres, the CUDA implementation replaces the object hierarchy with a **flat array of concrete `sphere` structures**.

Instead of:

```text
shared_ptr<hittable>
        ↓
virtual hit()
        ↓
dynamically allocated object
```

the GPU operates directly on contiguous sphere data.

This deliberately trades abstraction and extensibility for simpler control flow, better locality, and easier compiler optimization.

---

### Flattening recursion into iteration

The CPU implementation evaluates ray color recursively.

Each surface hit produces another ray and recursively evaluates the next bounce, with each level contributing an attenuation factor to the eventual returned color.

The CUDA implementation replaces this with an iterative bounce loop carrying an accumulated throughput value.

Conceptually, a recursive expression such as

```text
0.5 × (0.5 × (0.5 × sky))
```

becomes:

```text
attenuation = 1

for each bounce:
    attenuation *= 0.5

return attenuation * sky
```

The mathematical operation is equivalent, but the iterative form avoids recursive function calls and their associated per-thread stack and call overhead.

This transformation is one of the central ideas in the port: a recursive computation that accumulates a product can instead be represented explicitly as a loop with an accumulator.

---

### Per-thread randomness

The CPU implementation's random-number facilities cannot simply be used inside a CUDA kernel.

Instead, every rendering thread owns its own **cuRAND state**, initialized so that pixels receive independent pseudorandom sequences.

Each thread then consumes its state while generating:

* subpixel sample offsets
* random diffuse directions
* stochastic bounce paths

This removes shared RNG state and allows thousands of pixels to generate random samples concurrently.

---

## Optimization: Removing RNG Warp Divergence

After completing the initial CUDA port, the random-direction generator contained a source of avoidable control-flow divergence.

CUDA executes threads in groups of **32 called warps**. Threads within a warp execute instructions together. If different threads take different control-flow paths or execute loops for different numbers of iterations, some lanes may have to wait while others continue.

This is **warp divergence**.

### Initial implementation

The original `random_unit_vector` used rejection sampling.

Conceptually:

```text
repeat:
    generate random point in cube

until point lies inside unit sphere

normalize point
```

The number of iterations is random.

One thread might accept its first sample while another thread in the same warp could require several attempts. The warp cannot completely move past the loop until every active lane has finished.

Because this sampling operation occurs during ray scattering, the divergence can be repeated across many pixels and bounces.

---

### Divergence-free sampling

The rejection loop was replaced with an analytic construction.

A uniformly distributed point on the unit sphere can be generated from:

* a uniform `z ∈ [-1, 1]`
* a uniform azimuth `φ ∈ [0, 2π)`

with

```text
r = sqrt(1 - z²)

x = r cos(φ)
y = r sin(φ)
z = z
```

This produces a uniform point on the sphere without rejection sampling.

Every thread now performs a fixed amount of work:

```text
generate z
generate φ
compute r
compute x, y, z
```

The variable-length rejection loop disappears, eliminating warp divergence from this source.

---

## Measured Impact

The optimization reduced kernel time from approximately:

```text
18.5 ms → 15.2 ms
```

or roughly:

```text
18% faster
```

The improvement is meaningful but not enormous, which is consistent with the workload.

In this simple scene, many rays escape into the sky after relatively few bounces. `random_unit_vector` is therefore only one component of the total kernel runtime rather than the dominant operation.

A more enclosed or bounce-heavy scene would invoke the scattering code more frequently and could make this source of divergence more significant.

The important result is not simply the 18% number, but the optimization process:

```text
identify architecture-specific inefficiency
        ↓
change the algorithm
        ↓
benchmark before and after
        ↓
interpret the measured result
```

---

## CUDA Configuration

The CUDA implementation is compiled using:

```sh
-arch=sm_89
```

targeting the RTX 4090's Ada architecture.

The renderer currently launches **8×8 thread blocks**, giving 64 threads — two warps — per block.

Block-size tuning has not yet been systematically benchmarked, so this configuration should be considered the current implementation rather than an experimentally established optimum.

---

## Project Structure

```text
Raytrace.cpp    CPU path tracer
Raytracer.cu    CUDA path tracer
image.ppm       Rendered output
```

The CPU version acts as the reference implementation from which the CUDA version was derived.

---

## Limitations and Future Work

### Multithreaded CPU baseline

The current CPU renderer is single-threaded. Parallelizing the CPU implementation would provide a much stronger comparison against modern multicore CPU hardware.

This is the most important missing benchmark before making broader CPU-versus-GPU performance claims.

### End-to-end timing

The current CUDA benchmark isolates kernel execution.

A second benchmark measuring total wall-clock time — including CUDA initialization, allocation, data movement, rendering, and output — would complement the current throughput measurement.

### More complex geometry

The current renderer only supports spheres and performs intersection tests linearly.

A more complete renderer would add:

* triangle primitives
* mesh loading
* materials
* additional light sources
* bounding-volume hierarchies (BVHs)

A GPU BVH would also introduce a more challenging control-flow problem because different rays traverse different branches of the hierarchy.

### Profiling

The RNG optimization was motivated by the execution behavior of rejection sampling and validated experimentally.

A deeper optimization pass using Nsight Compute could measure:

* achieved occupancy
* warp execution efficiency
* branch efficiency
* instruction mix
* register pressure
* memory throughput
* scheduler stall reasons

Those measurements would provide a more empirical basis for choosing the next optimization target.

### Additional optimization targets

Potential experiments include:

* benchmarking alternative CUDA block dimensions
* constant-memory placement for small static scene data
* reducing divergence in the bounce loop
* reducing register pressure
* improving memory layout as scene complexity increases

Each optimization should be measured independently rather than assumed to improve performance.

---

## Build and Run

A C++ compiler is required for the CPU version and the CUDA toolkit is required for the GPU version.

### CPU

```sh
clang++ -O2 Raytrace.cpp -o rt
./rt
```

### CUDA

```sh
nvcc -ccbin g++-14 -arch=sm_89 Raytracer.cu -o rtgpu
./rtgpu
```

`-ccbin g++-14` selects the host compiler used by `nvcc`.

`-arch=sm_89` generates code targeting NVIDIA Ada GPUs such as the RTX 4090.

Both implementations write the rendered image to:

```text
image.ppm
```

The GPU implementation additionally reports kernel timing information to `stderr`.

The result can be viewed directly with:

```sh
feh image.ppm
```

or converted to PNG:

```sh
convert image.ppm image.png
```

---

## Why This Project

From a compiler perspective, the CPU-to-GPU port is also an exercise in **lowering a convenient high-level representation into one better suited to a constrained execution model**.

The original CPU implementation uses abstractions that make the program natural to write:

```text
recursive evaluation
virtual dispatch
dynamically referenced objects
serial loops
general-purpose randomness
```

The CUDA implementation makes several execution properties explicit:

```text
iteration
concrete data layouts
explicit parallelism
per-thread state
more uniform control flow
architecture-specific compilation
```

These are manual source-level transformations rather than transformations CUDA automatically performs for arbitrary CPU programs. However, the reasoning behind them closely mirrors concerns encountered in compiler optimization and GPU backend work: control-flow structure, data layout, call overhead, divergence, target architecture, and the mapping of high-level computation onto hardware execution units.

The project therefore became more than a renderer. It is an experiment in taking a correct high-level implementation, understanding where its abstractions conflict with the target architecture, lowering those abstractions deliberately, and validating the resulting optimization through measurement.
