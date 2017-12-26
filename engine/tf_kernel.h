#ifndef INTERACTIVE_KERNEL_H_
#define INTERACTIVE_KERNEL_H_

// Tensorflow Dependent Defines
#define COMPILER_MSVC
#define NOMINMAX
#define PROTOBUF_USE_DLLS
#define EIGEN_USE_THREADS

#include "tf_cuda.h"
#pragma warning(push, 0)
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#pragma warning(pop)

namespace TF = tensorflow;

namespace PLUGIN_NAMESPACE {

	void setup_kernels();

} // PLUGIN_NAMESPACE

template <typename Device, typename T>
struct InteractiveNormalsInputFunctor {
	cudaError_t operator()(const Device& d, int width, int height, size_t pitch, const void* in, T* out);
};

template <typename Device, typename T>
struct InteractiveDepthInputFunctor {
	cudaError_t operator()(const Device& d, int width, int height, size_t pitch, float min, float max, const void* in, T* out);
};

template <typename Device, typename T>
struct InteractiveOutputFunctor {
	cudaError_t operator()(const Device& d, int width, int height, size_t pitch, const T* in, void* out);
};

template <typename Device, typename T>
struct InteractiveDepthOutputFunctor {
	cudaError_t operator()(const Device& d, int width, int height, size_t pitch, float min, float max, const T* in, void* out);
};

template <typename Device, typename T>
class InteractiveNormalsInputOp : public TF::OpKernel {
public:
	explicit InteractiveNormalsInputOp(TF::OpKernelConstruction* context) : TF::OpKernel(context) {}

	void Compute(TF::OpKernelContext* context) override {
		// Grab the input tensor
		const TF::Tensor& input_tensor = context->input(0);

		// Set the allocated memory to be GPU based
		TF::AllocatorAttributes attributes;
		attributes.set_on_host(false);
		attributes.set_nic_compatible(false);
		attributes.set_gpu_compatible(true);

		_memory = PLUGIN_NAMESPACE::TFCuda::get_input_memory_pointer();
		_pitch = PLUGIN_NAMESPACE::TFCuda::get_pitch();

		// Create an output tensor and condition checking
		TF::Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
			&output_tensor, attributes));

		OP_REQUIRES(context, output_tensor->shape().dims() == 4,
			TF::errors::Unavailable("Interactive Input expects 4 dimensions (batch, width, height, channels)"));

		OP_REQUIRES(context, output_tensor->shape().dim_size(3) == 4,
			TF::errors::Unavailable("Interactive Input expects 4 channels"));

		OP_REQUIRES(context, _memory != nullptr,
			TF::errors::Unavailable("Could not get normals memory"));

		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
			TF::errors::InvalidArgument("Too many elements in tensor"));

		// Do the computation.
		cudaError_t result = InteractiveNormalsInputFunctor<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int>(input_tensor.shape().dim_size(1)),
			static_cast<int>(input_tensor.shape().dim_size(2)),
			_pitch,
			_memory,
			output_tensor->flat<T>().data());

		OP_REQUIRES(context, result == cudaSuccess, TF::errors::Internal("CUDA Error occured!"));
	}

private:
	size_t _pitch = 0;
	void *_memory = nullptr;
};

template <typename Device, typename T>
class InteractiveDepthInputOp : public TF::OpKernel {
public:
	explicit InteractiveDepthInputOp(TF::OpKernelConstruction* context) : TF::OpKernel(context) {}

	void Compute(TF::OpKernelContext* context) override {
		// Grab the input tensor
		const TF::Tensor& input_tensor = context->input(0);

		// Set the allocated memory to be GPU based
		TF::AllocatorAttributes attributes;
		attributes.set_on_host(false);
		attributes.set_nic_compatible(false);
		attributes.set_gpu_compatible(true);

		_near_range = PLUGIN_NAMESPACE::TFCuda::get_near_range();
		_far_range = PLUGIN_NAMESPACE::TFCuda::get_far_range();
		_memory = PLUGIN_NAMESPACE::TFCuda::get_depth_memory_pointer();
		_pitch = PLUGIN_NAMESPACE::TFCuda::get_pitch();

		// Create an output tensor and condition checking
		TF::Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
			&output_tensor, attributes));

		OP_REQUIRES(context, output_tensor->shape().dims() == 4,
			TF::errors::Unavailable("Interactive Input expects 4 dimensions (batch, width, height, channels)"));

		OP_REQUIRES(context, output_tensor->shape().dim_size(3) == 4,
			TF::errors::Unavailable("Interactive Input expects 4 channels"));

		OP_REQUIRES(context, _memory != nullptr,
			TF::errors::Unavailable("Could not get depth memory"));

		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
			TF::errors::InvalidArgument("Too many elements in tensor"));

		// Do the computation.
		cudaError_t result = InteractiveDepthInputFunctor<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int>(input_tensor.shape().dim_size(1)),
			static_cast<int>(input_tensor.shape().dim_size(2)),
			_pitch,
			_near_range,
			_far_range,
			_memory,
			output_tensor->flat<T>().data());

		OP_REQUIRES(context, result == cudaSuccess, TF::errors::Internal("CUDA Error occured!"));
	}

private:
	float _near_range = 0.1f;
	float _far_range = 1000.0f;
	size_t _pitch = 0;
	void *_memory = nullptr;
};

template <typename Device, typename T>
class InteractiveOutputOp : public TF::OpKernel {
public:
	explicit InteractiveOutputOp(TF::OpKernelConstruction* context) : TF::OpKernel(context) {}

	void Compute(TF::OpKernelContext* context) override {
		// Grab the input tensor
		const TF::Tensor& input_tensor = context->input(0);

		// Set the allocated memory to be GPU based
		TF::AllocatorAttributes attributes;
		attributes.set_on_host(false);
		attributes.set_nic_compatible(false);
		attributes.set_gpu_compatible(true);

		_memory = PLUGIN_NAMESPACE::TFCuda::get_output_memory_pointer();
		_pitch = PLUGIN_NAMESPACE::TFCuda::get_pitch();

		// Create an output tensor and condition checking
		TF::Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
			&output_tensor, attributes));

		OP_REQUIRES(context, input_tensor.shape().dims() == 4,
			TF::errors::Unavailable("Interactive Output expects 4 dimensions (batch, width, height, channels)"));

		OP_REQUIRES(context, output_tensor->shape().dim_size(3) == 4,
			TF::errors::Unavailable("Interactive Input expects 4 channels"));

		OP_REQUIRES(context, _memory != nullptr,
			TF::errors::Unavailable("Could not get texture memory"));

		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
			TF::errors::InvalidArgument("Too many elements in tensor"));

		// Do the computation.
		cudaError_t result = InteractiveOutputFunctor<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int>(input_tensor.shape().dim_size(1)),
			static_cast<int>(input_tensor.shape().dim_size(2)),
			_pitch,
			input_tensor.flat<T>().data(),
			_memory);

		OP_REQUIRES(context, result == cudaSuccess, TF::errors::Internal("CUDA Error occured!"));
	}

private:
	void *_memory = nullptr;
	size_t _pitch = 0;
};

template <typename Device, typename T>
class InteractiveDepthOutputOp : public TF::OpKernel {
public:
	explicit InteractiveDepthOutputOp(TF::OpKernelConstruction* context) : TF::OpKernel(context) {}

	void Compute(TF::OpKernelContext* context) override {
		// Grab the input tensor
		const TF::Tensor& input_tensor = context->input(0);

		// Set the allocated memory to be GPU based
		TF::AllocatorAttributes attributes;
		attributes.set_on_host(false);
		attributes.set_nic_compatible(false);
		attributes.set_gpu_compatible(true);

		_near_range = PLUGIN_NAMESPACE::TFCuda::get_near_range();
		_far_range = PLUGIN_NAMESPACE::TFCuda::get_far_range();
		_memory = PLUGIN_NAMESPACE::TFCuda::get_output_memory_pointer();
		_pitch = PLUGIN_NAMESPACE::TFCuda::get_pitch();

		// Create an output tensor and condition checking
		TF::Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
			&output_tensor, attributes));

		OP_REQUIRES(context, input_tensor.shape().dims() == 4,
			TF::errors::Unavailable("Interactive Output expects 4 dimensions (batch, width, height, channels)"));

		OP_REQUIRES(context, output_tensor->shape().dim_size(3) == 4,
			TF::errors::Unavailable("Interactive Input expects 4 channels"));

		OP_REQUIRES(context, _memory != nullptr,
			TF::errors::Unavailable("Could not get texture memory"));

		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
			TF::errors::InvalidArgument("Too many elements in tensor"));

		// Do the computation.
		cudaError_t result = InteractiveDepthOutputFunctor<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int>(input_tensor.shape().dim_size(1)),
			static_cast<int>(input_tensor.shape().dim_size(2)),
			_pitch,
			_near_range,
			_far_range,
			input_tensor.flat<T>().data(),
			_memory);

		OP_REQUIRES(context, result == cudaSuccess, TF::errors::Internal("CUDA Error occured!"));
	}

private:
	float _near_range = 0.1f;
	float _far_range = 1000.0f;
	void *_memory = nullptr;
	size_t _pitch = 0;
};

template <typename Device, typename T>
class InteractiveDebugPrintOp : public TF::OpKernel {
public:
	explicit InteractiveDebugPrintOp(TF::OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(TF::OpKernelContext* context) override {
		// Grab the input tensor
		const TF::Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<T>();

		// Create an output tensor
		TF::Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
		auto output = output_tensor->flat<T>();

		auto N = input.size();
		for (auto i = 0; i < N; i++) {
			output(i) = input(i);
		}
	}
};

#endif //INTERACTIVE_KERNEL_H_