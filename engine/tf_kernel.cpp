#ifdef GOOGLE_CUDA

#include "tf_kernel.h"

// Register the Interactive Ops and Kernels
REGISTER_OP("InteractiveInput")
.Input("interactive_input: float")
.Output("from_interactive: float")
.SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("InteractiveOutput")
.Input("to_interactive: float")
.Output("interactive_output: float")
.SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_KERNEL_BUILDER(Name("InteractiveInput").Device(TF::DEVICE_GPU), InteractiveInputOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("InteractiveOutput").Device(TF::DEVICE_GPU), InteractiveOutputOp<Eigen::GpuDevice, float>);

#endif  // GOOGLE_CUDA