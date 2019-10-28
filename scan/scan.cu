#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

inline static int ceilLog2(int32_t val) {
  assert(val > 0);

  int n = 0;
  while (val > (1 << n)) {
    n++;
  }

  return n;
}

void exclusive_scan_iterative(int logN, int* output) {
  // upsweep phase
  for (int d = 0; d < logN-2; ++d) {
    /*parallel*/for (int k = 0; k < (1 << (logN - d - 1)); ++k) {
      const int ix0 = (k << (d + 1)) + (1 << d) - 1;
      const int ix1 = ix0 + (1 << d);

      output[ix1] += output[ix0];
    }
  }

  // junction phase
  const int ix0 = (1 << (logN - 2)) - 1;
  const int ix1 = (1 << (logN - 1)) - 1;

  output[(1<<logN) - 1] = output[ix0] + output[ix1];
  output[ix1] = 0;

  // downsweep phase
  for (int d = logN-2; d >= 0; --d) {
    /*parallel*/for (int k = 0; k < (1 << (logN - d - 1)); ++k) {
      const int ix0 = (k << (d + 1)) + (1 << d) - 1;
      const int ix1 = ix0 + (1 << d);

      const int t = output[ix0];
      output[ix0] = output[ix1];
      output[ix1] += t;
    }
  }
}

__global__ void upsweep_kernel(int d, int* output) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  const int ix0 = (k << (d + 1)) + (1 << d) - 1;
  const int ix1 = ix0 + (1 << d);

  output[ix1] += output[ix0];
}

__global__ void junction_kernel(int* output, int logN) {
  const int ix0 = (1 << (logN - 2)) - 1;
  const int ix1 = (1 << (logN - 1)) - 1;

  output[(1 << logN) - 1] = output[ix0] + output[ix1];
  output[ix1] = 0;
}

__global__ void downsweep_kernel(int d, int* output) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  const int ix0 = (k << (d + 1)) + (1 << d) - 1;
  const int ix1 = ix0 + (1 << d);

  const int t = output[ix0];
  output[ix0] = output[ix1];
  output[ix1] += t;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
static void exclusive_scan(int logN, int* output) {
  for (int d = 0; d < logN - 2; ++d) {
    const int threads = 1 << (logN - d - 1);
    const int threadsPerBlock = std::min(threads, THREADS_PER_BLOCK);

    upsweep_kernel<<< threads/threadsPerBlock, threadsPerBlock >>> (d, output);
  }

  junction_kernel <<< 1,1 >>> (output, logN);

  for (int d = logN - 2; d >= 0; --d) {
    const int threads = 1 << (logN - d - 1);
    const int threadsPerBlock = std::min(threads, THREADS_PER_BLOCK);

    downsweep_kernel <<< threads / threadsPerBlock, threadsPerBlock >>> (d, output);
  }
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    const int N = end - inarray;
    const int logN = ceilLog2(N);
# if 0
    {
      assert(N == (1 << logN));
      memmove(resultarray, inarray, N * sizeof(int));

      const double startTime = CycleTimer::currentSeconds();
      exclusive_scan_iterative(logN, resultarray);
      const double endTime = CycleTimer::currentSeconds();

      return endTime - startTime;
    }
#endif
    int* device_result;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    cudaMalloc((void **)&device_result, sizeof(int) << logN);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, N * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(logN, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void equal_indices(const int* input, int* idx) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;

  idx[k] = input[k + 1] == input[k] ? 1 : 0;
//  printf("input[%i] = %i  idx[%i] = %i\n", k, input[k], k, idx[k]);
}

__global__ void equal_indices_ex(const int* input, int offset, int* idx) {
  const int k = offset  + blockIdx.x * blockDim.x + threadIdx.x;

  idx[k] = input[k + 1] == input[k] ? 1 : 0;
//  printf("input[%i] = %i  idx[%i] = %i\n", k, input[k], k, idx[k]);
}

static void mark_repeats(const int* input, int length, int* idx) {
  const int N = length - 1; // Last index must not be tested
  if (N >= THREADS_PER_BLOCK) {
    equal_indices << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (input, idx);
  }

  const int processed = N / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
  if (N > processed) {
    equal_indices_ex <<<1, N - processed >>> (input, processed, idx);
  }
}

__global__ void collect_kernel(const int* idx_sums, int offset, int* result) {
  const int k = offset+blockIdx.x * blockDim.x + threadIdx.x;

//  printf("idx_sums[%i] = %i\n", k, idx_sums[k]);

  int idx = idx_sums[k];
  if (idx_sums[k] != idx_sums[k+1]) {
    result[idx] = k;

//    printf("result[%i] = %i\n", idx, result[idx]);
  }
}

static void collect_indices(const int* idx_sums, int length, int* result) {
  const int N = length - 1; // Last index is used for result
  if (N >= THREADS_PER_BLOCK) {
    collect_kernel <<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (idx_sums, 0, result);
  }

  const int processed = N / THREADS_PER_BLOCK * THREADS_PER_BLOCK;
  if (N > processed) {
    collect_kernel <<<1, N - processed >>> (idx_sums, processed, result);
  }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* input, int length, int logN, int* scratch) {
  mark_repeats(input, length, scratch);
  exclusive_scan(logN, scratch);

  int repeats;
  cudaMemcpy(&repeats, scratch+length-1, sizeof(repeats), cudaMemcpyDeviceToHost);

  collect_indices(scratch, length, input);

  return repeats;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    const int logN = ceilLog2(length);

    int* data;    cudaMalloc(&data, length* sizeof(int));
    int* scratch; cudaMalloc(&scratch, sizeof(int) << logN);
    cudaMemcpy(data, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    const int repeats = find_repeats(data, length, logN, scratch);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = repeats;
    cudaMemcpy(output, data, repeats * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(data);
    cudaFree(scratch);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
