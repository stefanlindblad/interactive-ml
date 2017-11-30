#ifdef __CUDACC__

#define EIGEN_USE_GPU

#include "tf_kernel.h"

namespace TF = tensorflow;

template <typename T>
__global__ void InteractiveInputKernel(int width, int height, size_t pitch, const unsigned char* in, T* out) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned char *src;
	T *dest;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;


	// get a pointer to the pixel at (x,y)
	dest = (T *) (out + y*pitch) + 4 * x;
	src = (unsigned char *) (in + y*pitch) + 4 * x;
	dest[0] = ((float) src[0]) / 255.0f;
	dest[1] = ((float) src[1]) / 255.0f;
	dest[2] = ((float) src[2]) / 255.0f;
	dest[3] = ((float) src[3]) / 255.0f;
}

template <typename T>
__global__ void InteractiveOutputKernel(int width, int height, size_t pitch, const T* in, unsigned char* out) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	T *src;
	unsigned char *dest;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	src = (T *) (in + y*pitch) + 4 * x;
	dest = (unsigned char *) (out + y*pitch) + 4 * x;
	dest[0] = (unsigned char) (src[0] * 255.0f);
	dest[1] = (unsigned char) (src[1] * 255.0f);
	dest[2] = (unsigned char) (src[2] * 255.0f);
	dest[3] = (unsigned char) (src[3] * 255.0f);
}

template <typename T>
struct InteractiveInputFunctor<Eigen::GpuDevice, T> {
	void operator()(const Eigen::GpuDevice& d, int width, int height, size_t pitch, const void* in, T* out) {

		dim3 blockSize = dim3(16, 16);
		dim3 threadSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		InteractiveInputKernel<T><<<blockSize, threadSize>>>(width, height, pitch, (unsigned char *) in, out);
	}
};

template <typename T>
struct InteractiveOutputFunctor<Eigen::GpuDevice, T> {
	void operator()(const Eigen::GpuDevice& d, int width, int height, size_t pitch, const T* in, void* out) {
		dim3 blockSize = dim3(16, 16);
		dim3 threadSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		InteractiveOutputKernel<T><<<blockSize, threadSize>>>(width, height, pitch, in, (unsigned char *) out);
	}
};

template struct InteractiveInputFunctor<Eigen::GpuDevice, float>;
template struct InteractiveOutputFunctor<Eigen::GpuDevice, float>;

#endif  // __CUDACC__