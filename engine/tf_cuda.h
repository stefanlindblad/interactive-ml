#pragma once
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include <engine_plugin_api/plugin_api.h>

namespace PLUGIN_NAMESPACE
{
	struct ApiInterface
	{
		DataCompilerApi *_data_compiler;
		DataCompileParametersApi *_data_compile_parameters;
		ErrorApi *_error;
		LoggingApi *_logging;
		FileSystemApi *_file_system;
		LuaApi *_lua;
		RenderBufferApi *_render_buffer;
		RenderInterfaceApi *_render_interface;
		MeshObjectApi *_mesh;
		MaterialApi *_material;
		StreamCaptureApi *_capture;
		AllocatorApi *_allocator;
		AllocatorObject *_allocator_object;
		ResourceManagerApi *_resource_manager;
		ApplicationApi *_application;
		ApplicationOptionsApi *_options;
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
		static ApiInterface* get_api();
		static void set_api(ApiInterface *api);
	};
}
