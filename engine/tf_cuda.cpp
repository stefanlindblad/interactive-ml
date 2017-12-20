#include "tf_cuda.h"

namespace PLUGIN_NAMESPACE
{
	struct CUDA_transfer_data
	{
		void *_input_memory;
		void *_depth_memory;
		void *_output_memory;
		size_t _pitch;
		ApiInterface *_api;
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

	ApiInterface *TFCuda::get_api()
	{
		return instance_data._api;
	}

	void TFCuda::set_api(ApiInterface *api)
	{
		instance_data._api = api;
	}
}
