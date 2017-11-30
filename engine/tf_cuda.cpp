#include "tf_cuda.h"
#include <cuda_d3d11_interop.h>

namespace PLUGIN_NAMESPACE
{
	struct CUDA_transfer_data
	{
		void *_texture_memory;
		size_t _texture_pitch;
	};
	
	static CUDA_transfer_data instance_data;

	void *TFCuda::get_cuda_memory_pointer()
	{
		return instance_data._texture_memory;
	}
	
	void TFCuda::set_cuda_memory_pointer(void *pointer)
	{
		instance_data._texture_memory = pointer;
	}

	size_t TFCuda::get_cuda_pitch()
	{
		return instance_data._texture_pitch;
	}

	void TFCuda::set_cuda_pitch(size_t pitch)
	{
		instance_data._texture_pitch = pitch;
	}
}
