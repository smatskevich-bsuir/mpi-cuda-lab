#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/image_processing.cuh"
#include "../include/helper_image.h"

#define CH1_BLOCK_SIZE dim3(16, 16, 1)
#define CH1_WINDOW_SIZE dim3(4, 3, 1)

__device__ inline int x_gradient(u_char p11, u_char p12, u_char p13, u_char p21, u_char p22, u_char p23, u_char p31, u_char p32, u_char p33)
{
	return p11 + p21 + p31 - p13 - p23 - p33;
}

__device__ inline int y_gradient(u_char p11, u_char p12, u_char p13, u_char p21, u_char p22, u_char p23, u_char p31, u_char p32, u_char p33)
{
	return p11 + p12 + p13 - p31 - p32 - p33;
}

__device__ inline int prewitt_gradient(u_char p11, u_char p12, u_char p13, u_char p21, u_char p22, u_char p23, u_char p31, u_char p32, u_char p33)
{
	int gx = x_gradient(p11, p12, p13, p21, p22, p23, p31, p32, p33);
	int gy = y_gradient(p11, p12, p13, p21, p22, p23, p31, p32, p33);
	int sum = abs(gx) + abs(gy);
	sum = sum > 255 ? 255 : sum;
	sum = sum < 0 ? 0 : sum;
	return sum;
}

__global__ void prewitt_cylce_1ch(const u_char* data, u_char* res, const size_t width, const size_t height, const size_t pitch_in, const size_t pitch_out) 
{
	const dim3 offset_block(blockIdx.x * blockDim.x * (CH1_WINDOW_SIZE.x - 2), blockIdx.y * blockDim.y * (CH1_WINDOW_SIZE.y - 2));
	const dim3 offset_thread(threadIdx.x * (CH1_WINDOW_SIZE.x - 2), threadIdx.y * (CH1_WINDOW_SIZE.y - 2));

	//memory for block with align to 4 bytes
	const size_t shared_pixels_width = ((CH1_WINDOW_SIZE.x + (CH1_BLOCK_SIZE.x - 1) * (CH1_WINDOW_SIZE.x - 2) + sizeof(u_int32_t) - 1) / sizeof(u_int32_t)) * sizeof(u_int32_t);
	const size_t shared_pixels_height = CH1_WINDOW_SIZE.y + (CH1_BLOCK_SIZE.y - 1) * (CH1_WINDOW_SIZE.y - 2);
	__shared__ u_char shared_pixels[shared_pixels_height][shared_pixels_width];

	const size_t real_pixels_width = min(shared_pixels_width, width - offset_block.x);
	const size_t real_pixels_height = min(shared_pixels_height, height - offset_block.y);

	const size_t copy_threads_x = (real_pixels_width + sizeof(u_int32_t) - 1) / sizeof(u_int32_t);
	const size_t copy_threads_y = real_pixels_height;

	//use this and add additional reads for threadIdx.y = 0 and 1
	/*if(threadIdx.x < copy_threads_x && threadIdx.y < copy_threads_y)
		memcpy(shared_pixels[threadIdx.y] + threadIdx.x * sizeof(u_int32_t), data + (offset_block.y + threadIdx.y) * pitch_in + offset_block.x + threadIdx.x * sizeof(u_int32_t), sizeof(u_int32_t));*/

	//or this but with % and /
	const size_t idx = threadIdx.x + threadIdx.y * blockDim.x;
	const size_t read_y = idx / copy_threads_x;
	const size_t read_x = idx % copy_threads_x;
	if(read_x < copy_threads_x && read_y < copy_threads_y)
		memcpy(shared_pixels[read_y] + read_x * sizeof(u_int32_t), data + (offset_block.y + read_y) * pitch_in + offset_block.x + read_x * sizeof(u_int32_t), sizeof(u_int32_t));

	__syncthreads();

	
	//can calculate at least 3x3
	if(offset_block.x + offset_thread.x + 2 < width && offset_block.y + offset_thread.y + 2 < height)
	{
		//hardcode for WINDOW_SIZE dim3(4, 3, 1);
		u_int32_t words[3];
		memcpy(words, shared_pixels[offset_thread.y] + offset_thread.x, sizeof(u_int32_t));
		memcpy(words + 1, shared_pixels[offset_thread.y + 1] + offset_thread.x, sizeof(u_int32_t));
		memcpy(words + 2, shared_pixels[offset_thread.y + 2] + offset_thread.x, sizeof(u_int32_t));

		u_char out[2];
		out[0] = prewitt_gradient(words[0] >> 24 & 0xFF, words[0] >> 16 & 0xFF, words[0] >> 8 & 0xFF, words[1] >> 24 & 0xFF, words[1] >> 16 & 0xFF, words[1] >> 8 & 0xFF, words[2] >> 24 & 0xFF, words[2] >> 16 & 0xFF, words[2] >> 8 & 0xFF);
		out[1] = prewitt_gradient(words[0] >> 16 & 0xFF, words[0] >> 8 & 0xFF, words[0] & 0xFF, words[1] >> 16 & 0xFF, words[1] >> 8 & 0xFF, words[1] & 0xFF, words[2] >> 16 & 0xFF, words[2] >> 8 & 0xFF, words[2] & 0xFF);

		memcpy(res + offset_block.x + offset_thread.x + 1 + (offset_block.y + offset_thread.y + 1) * pitch_out, out, 2 * sizeof(u_char));
	}
}

void prewittGPU(const std::string& file)
{
	u_char* device_data, * device_res;
	u_char* input = 0, * output;
	u_int width, height, channels;
	size_t pitch_in, pitch_out;

	__loadPPM(file.c_str(), &input, &width, &height, &channels);
	output = (u_char*)malloc(width * height * channels * sizeof(u_char));

	cudaMallocPitch((void**)&device_data, &pitch_in, width * channels * sizeof(u_char), height);
	cudaMallocPitch((void**)&device_res, &pitch_out, width * channels * sizeof(u_char), height);
	cudaMemcpy2D(device_data, pitch_in, input, width * channels * sizeof(u_char), width * channels * sizeof(u_char), height, cudaMemcpyHostToDevice);

	switch (channels)
	{
		case 1:
			{
				dim3 real_block(CH1_BLOCK_SIZE.x * (CH1_WINDOW_SIZE.x - 2), CH1_BLOCK_SIZE.y * (CH1_WINDOW_SIZE.y - 2));
				dim3 grid_size((width - 2 + real_block.x - 1) / real_block.x, (height - 2 + real_block.y - 1) / real_block.y);

				prewitt_cylce_1ch << <grid_size, CH1_BLOCK_SIZE >> > (device_data, device_res, width * channels, height, pitch_in, pitch_out);
				break;
			}
		case 3:
		default:
			break;
	}

	cudaDeviceSynchronize();

	cudaMemcpy2D(output, width * channels * sizeof(u_char), device_res, pitch_out, width * channels * sizeof(u_char), height, cudaMemcpyDeviceToHost);
	cudaFree(device_data);
	cudaFree(device_res);

	cudaFreeHost(input);

	std::filesystem::path input_path(file);
	std::filesystem::path ouput_path(input_path.parent_path());
	ouput_path /= "result";
	ouput_path /= input_path.stem().string() + ".prewitt" + input_path.extension().string();

	__savePPM(ouput_path.c_str(), output, width, height, channels);
	free(output);
}