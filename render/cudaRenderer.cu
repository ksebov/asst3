#include <string>
#include <algorithm>
#include <cassert>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define BLOCKSIZE  512

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    CudaRenderer::Circle* circles;

    float* velocity;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    CudaRenderer::Circle* circle = cuConstRendererParams.circles;

    float* velocity = cuConstRendererParams.velocity;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int sIdx3 = 3 * sIdx;

    float cx = circle[fIdx].position[0];
    float cy = circle[fIdx].position[1];

    // update position
    circle[sIdx].position[0] += velocity[sIdx3+0] * dt;
    circle[sIdx].position[1] += velocity[sIdx3+1] * dt;

    // fire-work sparks
    float sx = circle[sIdx].position[0];
    float sy = circle[sIdx].position[1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * circle[fIdx].radius;
        float y = sinA * circle[fIdx].radius;

        circle[sIdx].position[0] = circle[fIdx].position[0] + x;
        circle[sIdx].position[1] = circle[fIdx].position[1] + y;
        circle[sIdx].position[2] = 0.0f;

        // travel scaled unit length 
        velocity[sIdx3 +0] = cosA/5.0;
        velocity[sIdx3 +1] = sinA/5.0;
        velocity[sIdx3 +2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float& radius = cuConstRendererParams.circles[index].radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius > cutOff) { 
        radius = 0.02f; 
    } else { 
        radius += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    CudaRenderer::Circle* circle = cuConstRendererParams.circles;
    float* velocity = cuConstRendererParams.velocity;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = circle[index].position[1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (circle[index].position[1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    circle[index].position[1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(circle[index].position[1] -oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        circle[index].position[1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = cuConstRendererParams.circles[index].position;
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.circles[index].radius;

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
struct SnowflakeShader {
  static __device__ __inline__ void
    shadePixel(int circleIndex, const float2 pixelCenter, float4* imagePtr) {
    const float3 p = *(float3*)cuConstRendererParams.circles[circleIndex].position;

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.circles[circleIndex].radius;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
      return;

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    float3 rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f - p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    float alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    float oneMinusAlpha = 1.f - alpha;

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;
  }
};

struct SimpleCircleShader {
  static __device__ __inline__ void
    shadePixel(int circleIndex, const float2 pixelCenter, float4* imagePtr) {
    const float3 p = *(float3*)cuConstRendererParams.circles[circleIndex].position;

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.circles[circleIndex].radius;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
      return;

    const float3 rgb = *(float3*)cuConstRendererParams.circles[circleIndex].color;

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = (rgb.x + existingColor.x)*.5f;
    newColor.y = (rgb.y + existingColor.y)*.5f;
    newColor.z = (rgb.z + existingColor.z)*.5f;
    newColor.w = .5f + existingColor.w;

    // global memory write
    *imagePtr = newColor;
  }
};

__device__ __inline__ void
shadePixel(int circleIndex, const float2 pixelCenter, float4* imagePtr) {
  if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
    SnowflakeShader::shadePixel(circleIndex, pixelCenter, imagePtr);
  } else {
    SimpleCircleShader::shadePixel(circleIndex, pixelCenter, imagePtr);
  }
}


// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    // read position and radius
    const float3 p   = *(float3*)cuConstRendererParams.circles[index].position;
    const float  rad = cuConstRendererParams.circles[index].radius;

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, imgPtr);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
}

CudaRenderer::~CudaRenderer() {

    delete image;
    delete [] velocity;

    cudaFree(cudaDeviceVelocity);
    cudaFree(cudaDeviceImageData);
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;

    float* position = nullptr;
    float* color = nullptr;
    float* radius = nullptr;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);

    circles_.clear();
    circles_.resize(numCircles + BLOCKSIZE); // Make sure there's a safety buffer for circle loop to overshoot.

    for (int c = 0; c < numCircles; ++c) {
      circles_[c].color[0] = color[0+3*c];
      circles_[c].color[1] = color[1+3*c];
      circles_[c].color[2] = color[2+3*c];

      circles_[c].position[0] = position[0 + 3 * c];
      circles_[c].position[1] = position[1 + 3 * c];
      circles_[c].position[2] = position[2 + 3 * c];

      circles_[c].radius = radius[c];
    }

    delete [] position;
    delete [] color;
    delete [] radius;
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaCircles, sizeof(Circle) * circles_.size());
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);

    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaCircles, circles_.data(), sizeof(cudaCircles[0]) * circles_.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.circles = cudaCircles;
    params.velocity = cudaDeviceVelocity;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

#define SCAN_BLOCK_DIM   BLOCKSIZE  // needed by sharedMemExclusiveScan implementation
#include "exclusiveScan.cu_inl"

__shared__ uint intersecting[BLOCKSIZE];
__shared__ uint intInd[BLOCKSIZE];
__shared__ uint toRender[BLOCKSIZE];
__shared__ uint prefixSumScratch[2 * BLOCKSIZE];

#include "circleBoxTest.cu_inl"

template <typename TShader>
__global__ void kernelRenderPixels() {
  const float boxL = float((blockIdx.x + 0) * blockDim.x) / cuConstRendererParams.imageWidth;
  const float boxB = float((blockIdx.y + 0) * blockDim.y) / cuConstRendererParams.imageHeight;
  const float boxR = float((blockIdx.x + 1) * blockDim.x) / cuConstRendererParams.imageWidth;
  const float boxT = float((blockIdx.y + 1) * blockDim.y) / cuConstRendererParams.imageHeight;

  const int linearIndex = threadIdx.y * blockDim.x + threadIdx.x;

  const int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

  float4 * imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * cuConstRendererParams.imageWidth + pixelX)]);
  const float2 pixelCenterNorm = make_float2(
    (pixelX + 0.5f) / cuConstRendererParams.imageWidth,
    (pixelY + 0.5f) / cuConstRendererParams.imageHeight);

  // Note: because of sharedMemExclusiveScan, last circle in the step is always ignored
  // That's why we advance by one step size less 1.
  for (int step = 0; step < cuConstRendererParams.numCircles; step+=(BLOCKSIZE-1)) {
    {
      const CudaRenderer::Circle& circle = cuConstRendererParams.circles[step + linearIndex];

      intersecting[linearIndex] =
        circleInBox(circle.position[0], circle.position[1], circle.radius, boxL, boxR, boxT, boxB);
    } __syncthreads();

    {
      sharedMemExclusiveScan(linearIndex, intersecting, intInd, prefixSumScratch, BLOCKSIZE);
    } __syncthreads();

    const int cRendered = intInd[BLOCKSIZE-1];
    // if(linearIndex==0) printf("cRendered: %i\n", cRendered);
    if (0 == cRendered) {
      continue;
    }

    {
      int idx = intInd[linearIndex];
      if (idx != intInd[linearIndex + 1]) {
        toRender[idx] = linearIndex;
      }
    } __syncthreads();

    {
      for (int circle = 0; circle < cRendered; ++circle) {
        const int index = step + toRender[circle];
        TShader::shadePixel(index, pixelCenterNorm, imgPtr);
      }
    } __syncthreads();
  }
}

void
CudaRenderer::render() {

#if 1
  constexpr int blockWidth = 32; assert(BLOCKSIZE % blockWidth == 0);
  dim3 blockDim(blockWidth, BLOCKSIZE/ blockWidth);
  assert(image && image->width % blockDim.x == 0 && image->height % blockDim.y == 0);
  dim3 gridDim(image->width / blockDim.x, image->height / blockDim.y);

  if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
    kernelRenderPixels<SnowflakeShader> <<<gridDim, blockDim >>> ();
  }
  else {
    kernelRenderPixels<SimpleCircleShader> <<<gridDim, blockDim >>> ();
  }

#else
  // 256 threads per block is a healthy number
  dim3 blockDim(BLOCKSIZE, 1);
  dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

  kernelRenderCircles << <gridDim, blockDim >> > ();
#endif
  cudaDeviceSynchronize();
}
