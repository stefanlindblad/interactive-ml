#pragma once

#include "tf_settings.h"
#include "tf_lua.h"
#include "tf_kernel.h"
#include "tf_cuda.h"
#include <engine_plugin_api/plugin_api.h>
#include <plugin_foundation/vector2.h>
#include <plugin_foundation/string.h>

// D3D11 and CUDA Headers
#include <d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

namespace PLUGIN_NAMESPACE
{
	namespace SPF = stingray_plugin_foundation;
	namespace TF = tensorflow;

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
		CApi *_c;
	};

	class TFPlugin
	{
	public:
		static ApiInterface &get_api();
		static SPF::ApiAllocator &get_allocator();
		static void setup_plugin(GetApiFunction get_engine_api);
		static void update_plugin(float dt);
		static void shutdown_plugin();
		static void setup_data_compiler(GetApiFunction get_engine_api);
		static void shutdown_data_compiler();
		static const char *get_name();
		static int can_refresh(uint64_t type);
		static TF::Status read_tf_graph(const std::string &path, unsigned mode, TF::GraphDef *def);
		static void end_tf_execution();
		static void run_tf_graph(const char *graph_name, const char *node_name, unsigned iterations);
		static bool getLastCudaError(const char *errorMessage, const char *file, const int line);
		static void render(RenderDevicePluginArguments *arguments);
		static void end_frame();
		static void* get_render_env();
	};
}
