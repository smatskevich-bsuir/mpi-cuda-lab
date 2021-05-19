#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_image.h"

void prewittGPU(const std::string& file); 