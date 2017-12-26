#pragma once
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

namespace PLUGIN_NAMESPACE
{
	struct CUDA_transfer_data
	{
		void *_input_memory;
		void *_depth_memory;
		void *_output_memory;
		size_t _pitch;
		float _near_range = 0.1f;
		float _far_range = 1000.0f;
	};

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
		static float get_near_range();
		static void set_near_range(float near_range);
		static float get_far_range();
		static void set_far_range(float far_range);
	};
}
