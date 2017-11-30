#pragma once

// Tensorflow Dependent Defines
#define COMPILER_MSVC
#define NOMINMAX
#define PROTOBUF_USE_DLLS
#define EIGEN_USE_THREADS

// Disabling all the warnings from Tensorflow
#pragma warning(push, 0)
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/load_library.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/util/port.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/c/c_api.h"
#pragma warning(pop)

#pragma warning( disable : 4700 )