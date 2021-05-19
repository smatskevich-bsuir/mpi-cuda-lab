#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/image_processing.cuh"
#include "../include/helper_image.h"

const size_t BLOCK_SIZE = 16;

__device__ int xGradient(const u_char* image, size_t x, size_t y, size_t ch, size_t width, size_t height, size_t channels)
{
	return image[((y - 1) * width + x - 1) * channels + ch] +
		image[(y * width + x - 1) * channels + ch] +
		image[((y + 1) * width + x - 1) * channels + ch] -
		image[((y - 1) * width + x + 1) * channels + ch] -
		image[(y * width + x + 1) * channels + ch] -
		image[((y + 1) * width + x + 1) * channels + ch];
}

__device__ int yGradient(const u_char* image, size_t x, size_t y, size_t ch, size_t width, size_t height, size_t channels)
{
	return image[((y - 1) * width + x - 1) * channels + ch] +
		image[((y - 1) * width + x) * channels + ch] +
		image[((y - 1) * width + x + 1) * channels + ch] -
		image[((y + 1) * width + x - 1) * channels + ch] -
		image[((y + 1) * width + x) * channels + ch] -
		image[((y + 1) * width + x + 1) * channels + ch];
}

__global__ void prewitt_cylce(u_char* data, u_char* res, size_t width, size_t height, size_t channels) {
	size_t offset_x = blockIdx.x * (blockDim.x - 2);
	size_t offset_y = blockIdx.y * (blockDim.y - 2);

	if (threadIdx.x + offset_x >= width)
		return;

	if (threadIdx.y + offset_y >= height)
		return;

	//skip edges
	if (threadIdx.x == 0)
		return;
	if (threadIdx.x == blockDim.x - 1)
		return;
	if (threadIdx.y == 0)
		return;
	if (threadIdx.y == blockDim.y - 1)
		return;
	if (threadIdx.x + offset_x + 1 >= width)
		return;
	if (threadIdx.y + offset_y + 1 >= height)
		return;

	for (size_t ch = 0; ch < channels; ch++)
	{
		int gx = xGradient(data, threadIdx.x + offset_x, threadIdx.y + offset_y, ch, width, height, channels);
		int gy = yGradient(data, threadIdx.x + offset_x, threadIdx.y + offset_y, ch, width, height, channels);
		int sum = abs(gx) + abs(gy);
		sum = sum > 255 ? 255 : sum;
		sum = sum < 0 ? 0 : sum;
		res[((threadIdx.y + offset_y) * width + threadIdx.x + offset_x) * channels + ch] = sum;
	}
}

void prewittGPU(const std::string& file)
{
	u_char* device_data, * device_res;
	u_char* input = 0, * output;
	u_int width, height, channels;

	__loadPPM(file.c_str(), &input, &width, &height, &channels);
	output = (u_char*)malloc(width * height * channels * sizeof(u_char));

	cudaMalloc((void**)&device_data, width * height * sizeof(u_char));
	cudaMalloc((void**)&device_res, width * height * sizeof(u_char));
	cudaMemcpy(device_data, input, width * height * sizeof(u_char), cudaMemcpyHostToDevice);

	dim3 real_block(BLOCK_SIZE - 2, BLOCK_SIZE - 2);
	dim3 grid_size((width - 2 + real_block.x - 1) / real_block.x, (height - 2 + real_block.y - 1) / real_block.y);

	prewitt_cylce << <grid_size, dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (device_data, device_res, width, height, channels);
	cudaDeviceSynchronize();

	cudaMemcpy(output, device_res, width * height * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaFree(device_data);
	cudaFree(device_res);

	std::filesystem::path input_path(file);
	std::filesystem::path ouput_path(input_path.parent_path());
	ouput_path /= "result";
	ouput_path /= input_path.stem().string() + ".prewitt" + input_path.extension().string();

	__savePPM(ouput_path.c_str(), output, width, height, channels);
}