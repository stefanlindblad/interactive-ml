#pragma once
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

namespace PLUGIN_NAMESPACE
{
	class TFCuda
	{
	public:
		static void *get_cuda_memory_pointer();
		static void set_cuda_memory_pointer(void *pointer);
		static size_t get_cuda_pitch();
		static void set_cuda_pitch(size_t pitch);
	};
}
