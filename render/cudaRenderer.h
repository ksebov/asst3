#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#include <vector>

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"


class CudaRenderer : public CircleRenderer {

public:
    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    struct Circle {
      float position[3] = {-1,-1, -1};
      float radius = 0;
      float color   [3] = {0,0,0};
      float _reserved = 0;
    };

private:
  Image* image = nullptr;
  SceneName sceneName;

  int numCircles = 0;
  std::vector<Circle> circles_;
  float* velocity = nullptr;

  Circle* cudaCircles = nullptr;
  float* cudaDeviceVelocity = nullptr;

  float* cudaDeviceImageData = nullptr;
};


#endif
