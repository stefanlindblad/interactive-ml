#pragma once
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

namespace PLUGIN_NAMESPACE
{
	class TFCuda
	{
	public:
		static void *get_input_memory_pointer();
		static void set_input_memory_pointer(void *pointer);
		static void *get_depth_memory_pointer();
		static void set_depth_memory_pointer(void *pointer);
		static void *get_output_memory_pointer();
		static void set_output_memory_pointer(void *pointer);
		static size_t get_pitch();
		static void set_pitch(size_t pitch);
	};
}
