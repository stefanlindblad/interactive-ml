#ifdef GOOGLE_CUDA

#include "tf_kernel.h"

// Register the Interactive Ops and Kernels
REGISTER_OP("InteractiveNormalsInput")
.Input("interactive_input: float")
.Output("from_interactive: float")
.SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("InteractiveDepthInput")
.Input("interactive_input: float")
.Output("from_interactive: float")
.SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("InteractiveOutput")
.Input("to_interactive: float")
.Output("interactive_output: float")
.SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_KERNEL_BUILDER(Name("InteractiveNormalsInput").Device(TF::DEVICE_GPU), InteractiveNormalsInputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("InteractiveDepthInput").Device(TF::DEVICE_GPU), InteractiveDepthInputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("InteractiveOutput").Device(TF::DEVICE_GPU), InteractiveOutputOp<Eigen::GpuDevice, float>);

#endif  // GOOGLE_CUDA