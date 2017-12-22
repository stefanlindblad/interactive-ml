#ifdef __CUDACC__

#define EIGEN_USE_GPU

#include "tf_kernel.h"

namespace TF = tensorflow;

template <typename T>
__global__ void InteractiveNormalsInputKernel(int width, int height, size_t pitch, const unsigned char* in, T* out) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned char *src;
	T *dest;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	src = (unsigned char *)(in + y*pitch) + 4 * x;
	dest = (T *)(out + y*pitch) + 4 * x;

	dest[0] = ((float)src[0]) / 255.0f;
	dest[1] = ((float)src[1]) / 255.0f;
	dest[2] = ((float)src[2]) / 255.0f;
	dest[3] = ((float)src[3]) / 255.0f;
}

template <typename T>
__global__ void InteractiveDepthInputKernel(int width, int height, size_t pitch, float min, float max, const float* in, T* out) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *src;
	T *dest;
	float range = max - min;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	src = (float *)(in + y*pitch) + 4 * x;
	dest = (T *)(out + y*pitch) + 4 * x;

	dest[0] = ((src[0] - min) / range);
	dest[1] = ((src[1] - min) / range);
	dest[2] = ((src[2] - min) / range);
	dest[3] = ((src[3] - min) / range);
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
__global__ void InteractiveDepthOutputKernel(int width, int height, size_t pitch, float min, float max, const T* in, float* out) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	T *src;
	float *dest;
	float range = max - min;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	src = (T *)(in + y*pitch) + 4 * x;
	dest = (float *)(out + y*pitch) + 4 * x;

	dest[0] = (src[0] * range) + min;
	dest[1] = (src[1] * range) + min;
	dest[2] = (src[2] * range) + min;
	dest[3] = (src[3] * range) + min;
}

template <typename T>
struct InteractiveNormalsInputFunctor<Eigen::GpuDevice, T> {
	void operator()(const Eigen::GpuDevice& d, int width, int height, size_t pitch, const void* in, T* out) {
		dim3 blockSize = dim3(width, height);
		dim3 threadSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		InteractiveNormalsInputKernel<T><<<blockSize, threadSize>>>(width, height, pitch, (const unsigned char *) in, out);
		cudaError_t err = cudaGetLastError();
	}
};

template <typename T>
struct InteractiveDepthInputFunctor<Eigen::GpuDevice, T> {
	void operator()(const Eigen::GpuDevice& d, int width, int height, size_t pitch, float min, float max, const void* in, T* out) {
		dim3 blockSize = dim3(width, height);
		dim3 threadSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		InteractiveDepthInputKernel<T><<<blockSize, threadSize>>>(width, height, pitch, min, max, (const float *) in, out);
		cudaError_t err = cudaGetLastError();
	}
};

template <typename T>
struct InteractiveOutputFunctor<Eigen::GpuDevice, T> {
	void operator()(const Eigen::GpuDevice& d, int width, int height, size_t pitch, const T* in, void* out) {
		dim3 blockSize = dim3(width, height);
		dim3 threadSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		InteractiveOutputKernel<T><<<blockSize, threadSize>>>(width, height, pitch, in, (unsigned char *) out);
		cudaError_t err = cudaGetLastError();
	}
};

template <typename T>
struct InteractiveDepthOutputFunctor<Eigen::GpuDevice, T> {
	void operator()(const Eigen::GpuDevice& d, int width, int height, size_t pitch, float min, float max, const T* in, void* out) {
		dim3 blockSize = dim3(width, height);
		dim3 threadSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
		InteractiveDepthOutputKernel<T> << <blockSize, threadSize >> >(width, height, pitch, min, max, in, (float *) out);
		cudaError_t err = cudaGetLastError();
	}
};

template struct InteractiveNormalsInputFunctor<Eigen::GpuDevice, float>;
template struct InteractiveDepthInputFunctor<Eigen::GpuDevice, float>;
template struct InteractiveOutputFunctor<Eigen::GpuDevice, float>;
template struct InteractiveDepthOutputFunctor<Eigen::GpuDevice, float>;

#endif  // __CUDACC__