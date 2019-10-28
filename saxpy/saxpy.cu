#include <stdio.h>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if (index < N)
       result[index] = alpha * x[index] + y[index];
}


// saxpyCuda --
//
// This function is regular C code running on the CPU.  It allocates
// memory on the GPU using CUDA API functions, uses CUDA API functions
// to transfer data from the CPU's memory address space to GPU memory
// address space, and launches the CUDA kernel function on the GPU.
template <typename T>
class auto_cuda_ptr : public std::unique_ptr<float, decltype(&cudaFree)> {
public:
  auto_cuda_ptr() : unique_ptr(nullptr, &cudaFree) {}

  explicit auto_cuda_ptr(int N)
  : unique_ptr(nullptr, &cudaFree) {
    T* p; cudaMalloc(&p, N * sizeof(T));
    reset(p);
  }
};


void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    // must read both input arrays (xarray and yarray) and write to
    // output array (resultarray)
    const auto GBPerSec = [=](float sec) {
      return static_cast<float>(N / (1024. * 1024. * 1024.) / sec * sizeof(float) * 3);
    };

    // compute number of blocks and threads per block.  In this
    // application we've hardcoded thread blocks to contain 512 CUDA
    // threads.
    const int threadsPerBlock = 512;

    // Notice the round up here.  The code needs to compute the number
    // of threads blocks needed such that there is one thread per
    // element of the arrays.  This code is written to work for values
    // of N that are not multiples of threadPerBlock.
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // These are pointers that will be pointers to memory allocated
    // *one the GPU*.  You should allocate these pointers via
    // cudaMalloc.  You can access the resulting buffers from CUDA
    // device kernel code (see the kernel function saxpy_kernel()
    // above) but you cannot access the contents these buffers from
    // this thread. CPU threads cannot issue loads and stores from GPU
    // memory!

    auto_cuda_ptr<float> device_x{N};
    auto_cuda_ptr<float> device_y{N};
    auto_cuda_ptr<float> device_result{N};

    const double overallStart = CycleTimer::currentSeconds();

    cudaMemcpy(device_x.get(), xarray, N * sizeof(xarray[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y.get(), yarray, N * sizeof(yarray[0]), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run CUDA kernel. (notice the <<< >>> brackets indicating a CUDA
    // kernel launch) Execution on the GPU occurs here.
    const double kernelStart = CycleTimer::currentSeconds();
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x.get(), device_y.get(), device_result.get());
    cudaDeviceSynchronize();
    const auto kernelEnd = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result.get(),  N * sizeof(resultarray[0]), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    const double overallEnd = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
		errCode, cudaGetErrorString(errCode));
    }

    const double overallDuration = overallEnd - overallStart;
    const double kernelDuration = kernelEnd - kernelStart;

    printf("Effective BW by CUDA saxpy kernel: %.3f ms\t total: %.3f ms\t[%.3f GB/s]\n",
      1000.f * kernelDuration,
      1000.f * overallDuration,
      GBPerSec(overallDuration - kernelDuration));

}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
