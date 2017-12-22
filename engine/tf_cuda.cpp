#include "tf_cuda.h"

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
	
	static CUDA_transfer_data instance_data;

	void *TFCuda::get_input_memory_pointer()
	{
		return instance_data._input_memory;
	}

	void TFCuda::set_input_memory_pointer(void *pointer)
	{
		instance_data._input_memory = pointer;
	}

	void *TFCuda::get_depth_memory_pointer()
	{
		return instance_data._depth_memory;
	}

	void TFCuda::set_depth_memory_pointer(void *pointer)
	{
		instance_data._depth_memory = pointer;
	}

	void *TFCuda::get_output_memory_pointer()
	{
		return instance_data._output_memory;
	}

	void TFCuda::set_output_memory_pointer(void *pointer)
	{
		instance_data._output_memory = pointer;
	}

	size_t TFCuda::get_pitch()
	{
		return instance_data._pitch;
	}

	void TFCuda::set_pitch(size_t pitch)
	{
		instance_data._pitch = pitch;
	}

	float TFCuda::get_near_range()
	{
		return instance_data._near_range;
	}

	void TFCuda::set_near_range(float near_range)
	{
		instance_data._near_range = near_range;
	}

	float TFCuda::get_far_range()
	{
		return instance_data._far_range;
	}

	void TFCuda::set_far_range(float far_range)
	{
		instance_data._far_range = far_range;
	}
}
