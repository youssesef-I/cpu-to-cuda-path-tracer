CXX := clang++
NVCC := nvcc

HOST_CXX ?= g++-14
ARCH ?= sm_89

CXXFLAGS := -O2
NVCCFLAGS := -O2 -arch=$(ARCH) -ccbin $(HOST_CXX)

CPU_TARGET := rt
GPU_TARGET := rtgpu
GPU_PRECISE_TARGET := rtgpu-precise
GPU_PROFILE_TARGET := rtgpu-profile

CPU_SRC := Raytrace.cpp
GPU_SRC := Raytracer.cu

.PHONY: all cpu gpu precise profile clean

all: $(CPU_TARGET) $(GPU_TARGET)

cpu: $(CPU_TARGET)

gpu: $(GPU_TARGET)

precise: $(GPU_PRECISE_TARGET)

profile: $(GPU_PROFILE_TARGET)

$(CPU_TARGET): $(CPU_SRC)
	$(CXX) $(CXXFLAGS) $(CPU_SRC) -o $(CPU_TARGET)

# Final optimized GPU build.
$(GPU_TARGET): $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(GPU_SRC) -o $(GPU_TARGET)

# Precise-math comparison build.
$(GPU_PRECISE_TARGET): $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) $(GPU_SRC) -o $(GPU_PRECISE_TARGET)

# Optimized build with source-line information for Nsight Compute.
$(GPU_PROFILE_TARGET): $(GPU_SRC)
	$(NVCC) $(NVCCFLAGS) --use_fast_math -lineinfo $(GPU_SRC) -o $(GPU_PROFILE_TARGET)

clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET) $(GPU_PRECISE_TARGET) \
	      $(GPU_PROFILE_TARGET) rtgpu-fastmath rtgpu-fastmath-profile \
	      *.o image.ppm
