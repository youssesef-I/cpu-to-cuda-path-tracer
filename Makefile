CXX := clang++
NVCC := nvcc

CXXFLAGS := -O2
NVCCFLAGS := -O2 -arch=sm_89 -ccbin g++-14

CPU_SRC := Raytrace.cpp
GPU_SRC := Raytracer.cu

CPU_BIN := rt
GPU_BIN := rtgpu

.PHONY: all cpu gpu run-cpu run-gpu clean

all: cpu gpu

cpu:
	$(CXX) $(CXXFLAGS) $(CPU_SRC) -o $(CPU_BIN)

gpu:
	$(NVCC) $(NVCCFLAGS) $(GPU_SRC) -o $(GPU_BIN)

run-cpu: cpu
	./$(CPU_BIN)

run-gpu: gpu
	./$(GPU_BIN)

clean:
	rm -f $(CPU_BIN) $(GPU_BIN) *.o *.ppm
