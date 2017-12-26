#include "tf_kernel.h"

namespace PLUGIN_NAMESPACE {

	void setup_kernels()
	{
		REGISTER_OP("InteractiveNormalsInput")
			.Input("interactive_input: float")
			.Output("from_interactive: float")
			.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			c->set_output(0, c->input(0));
			return TF::Status::OK();
		});

		REGISTER_OP("InteractiveDepthInput")
			.Input("interactive_input: float")
			.Output("from_interactive: float")
			.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			c->set_output(0, c->input(0));
			return TF::Status::OK();
		});

		REGISTER_OP("InteractiveOutput")
			.Input("to_interactive: float")
			.Output("interactive_output: float")
			.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			c->set_output(0, c->input(0));
			return TF::Status::OK();
		});

		REGISTER_OP("InteractiveDepthOutput")
			.Input("to_interactive: float")
			.Output("interactive_output: float")
			.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			c->set_output(0, c->input(0));
			return TF::Status::OK();
		});

		REGISTER_OP("InteractiveDebugPrint")
			.Input("to_print: float")
			.Output("printed: float")
			.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			c->set_output(0, c->input(0));
			return TF::Status::OK();
		});

		REGISTER_KERNEL_BUILDER(Name("InteractiveNormalsInput").Device(TF::DEVICE_GPU), InteractiveNormalsInputOp<Eigen::GpuDevice, float>);
		REGISTER_KERNEL_BUILDER(Name("InteractiveDepthInput").Device(TF::DEVICE_GPU), InteractiveDepthInputOp<Eigen::GpuDevice, float>);
		REGISTER_KERNEL_BUILDER(Name("InteractiveOutput").Device(TF::DEVICE_GPU), InteractiveOutputOp<Eigen::GpuDevice, float>);
		REGISTER_KERNEL_BUILDER(Name("InteractiveDepthOutput").Device(TF::DEVICE_GPU), InteractiveDepthOutputOp<Eigen::GpuDevice, float>);
		REGISTER_KERNEL_BUILDER(Name("InteractiveDebugPrint").Device(TF::DEVICE_CPU), InteractiveDebugPrintOp<Eigen::ThreadPoolDevice, float>);
	}
} // PLUGIN_NAMESPACE

#ifdef GOOGLE_CUDA

REGISTER_OP("InteractiveNormalsInput")
.Input("interactive_input: float")
.Output("from_interactive: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return TF::Status::OK();
});

REGISTER_OP("InteractiveDepthInput")
.Input("interactive_input: float")
.Output("from_interactive: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return TF::Status::OK();
});

REGISTER_OP("InteractiveOutput")
.Input("to_interactive: float")
.Output("interactive_output: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return TF::Status::OK();
});

REGISTER_OP("InteractiveDepthOutput")
.Input("to_interactive: float")
.Output("interactive_output: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return TF::Status::OK();
});

REGISTER_OP("InteractiveDebugPrint")
.Input("to_print: float")
.Output("printed: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return TF::Status::OK();
});

REGISTER_KERNEL_BUILDER(Name("InteractiveNormalsInput").Device(TF::DEVICE_GPU), InteractiveNormalsInputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("InteractiveDepthInput").Device(TF::DEVICE_GPU), InteractiveDepthInputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("InteractiveOutput").Device(TF::DEVICE_GPU), InteractiveOutputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("InteractiveDepthOutput").Device(TF::DEVICE_GPU), InteractiveDepthOutputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("InteractiveDebugPrint").Device(TF::DEVICE_CPU), InteractiveDebugPrintOp<Eigen::ThreadPoolDevice, float>);

#endif  // GOOGLE_CUDA