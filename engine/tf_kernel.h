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

template <typename Device, typename T>
struct InteractiveInputFunctor {
	void operator()(const Device& d, int width, int height, size_t pitch, const void* in, T* out);
};

template <typename Device, typename T>
struct InteractiveOutputFunctor {
	void operator()(const Device& d, int width, int height, size_t pitch, const T* in, void* out);
};

template <typename Device, typename T>
class InteractiveInputOp : public TF::OpKernel {
public:
	explicit InteractiveInputOp(TF::OpKernelConstruction* context) : TF::OpKernel(context) {}

	void Compute(TF::OpKernelContext* context) override {
		// Grab the input tensor
		const TF::Tensor& input_tensor = context->input(0);

		// Set the allocated memory to be GPU based
		TF::AllocatorAttributes attributes;
		attributes.set_on_host(false);
		attributes.set_nic_compatible(false);
		attributes.set_gpu_compatible(true);

		_pitch = PLUGIN_NAMESPACE::TFCuda::get_cuda_pitch();
		_memory = PLUGIN_NAMESPACE::TFCuda::get_cuda_memory_pointer();

		// Create an output tensor and condition checking
		TF::Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
			&output_tensor, attributes));

		OP_REQUIRES(context, output_tensor->shape().dims() == 4,
			TF::errors::Unavailable("Interactive Input expects 4 dimensions (batch, width, height, channels)"));

		OP_REQUIRES(context, _memory != nullptr,
			TF::errors::Unavailable("Could not get texture memory"));

		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
			TF::errors::InvalidArgument("Too many elements in tensor"));

		// Do the computation.
		InteractiveInputFunctor<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int>(input_tensor.shape().dim_size(1)),
			static_cast<int>(input_tensor.shape().dim_size(2)),
			_pitch,
			_memory,
			output_tensor->flat<T>().data());
	}

private:
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

		_memory = PLUGIN_NAMESPACE::TFCuda::get_cuda_memory_pointer();
		_pitch = PLUGIN_NAMESPACE::TFCuda::get_cuda_pitch();

		// Create an output tensor and condition checking
		TF::Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
			&output_tensor, attributes));

		OP_REQUIRES(context, input_tensor.shape().dims() == 4,
			TF::errors::Unavailable("Interactive Output expects 4 dimensions (batch, width, height, channels)"));

		OP_REQUIRES(context, _memory != nullptr,
			TF::errors::Unavailable("Could not get texture memory"));

		OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
			TF::errors::InvalidArgument("Too many elements in tensor"));


		// Do the computation.
		InteractiveOutputFunctor<Device, T>()(
			context->eigen_device<Device>(),
			static_cast<int>(input_tensor.shape().dim_size(1)),
			static_cast<int>(input_tensor.shape().dim_size(2)),
			_pitch,
			input_tensor.flat<T>().data(),
			_memory);
	}

private:
	void *_memory = nullptr;
	size_t _pitch = 0;
};

#endif //INTERACTIVE_KERNEL_H_