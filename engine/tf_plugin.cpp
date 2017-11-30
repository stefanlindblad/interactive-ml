#include "tf_plugin.h"

namespace PLUGIN_NAMESPACE
{
	//#define WAITFORDEBUGGER
	#define checkCUDAError(msg) if(getLastCudaError (msg, __FILE__, __LINE__)) return
	#define MEASURE_TIME

	namespace SPF = stingray_plugin_foundation;
	namespace TF = tensorflow;

	// Typedefs to use the high precision time measure clock
	typedef std::ratio<1l, 1000000000000l> pico;
	typedef std::chrono::duration<long long, pico> picoseconds;
	typedef std::ratio<1l, 1000000l> micro;
	typedef std::chrono::duration<long long, micro> microseconds;

	// Checks to see if the plugin apis have been properly initialized
	bool _compiler_api_initialized = false;
	bool _game_api_initialized = false;

	// Structure to define a tensorflow session, only one is supported right now
	struct Graph_Execution_Session
	{
		unsigned texture_width;
		unsigned texture_height;
		unsigned iterations_done;
		unsigned iterations_max;
		uint32_t override_texture_id;
		std::string output_node_name;
		cudaArray *cuda_array = nullptr;
		cudaGraphicsResource *cuda_resource = nullptr;
		ID3D11Texture2D *cuda_texture = nullptr;
		TF::Tensor *zero_input = nullptr;
		TF::Session *tf_session = nullptr;
		TF::GraphDef tf_graph;
	};

	static Graph_Execution_Session *session = nullptr;

	void wait_for_debugger()
	{
	#ifdef WAITFORDEBUGGER
		while( !IsDebuggerPresent() )
				Sleep( 100 ); 
	#endif
	}

	ApiInterface _api;
	ApiInterface& TFPlugin::get_api()
	{
		return _api;
	}

	SPF::ApiAllocator _tensorflow_allocator = SPF::ApiAllocator(nullptr, nullptr);
	SPF::ApiAllocator& TFPlugin::get_allocator()
	{
		return _tensorflow_allocator;
	}

	void init_compiler_api(GetApiFunction get_engine_api)
	{
		_api._data_compiler = (DataCompilerApi *)get_engine_api(DATA_COMPILER_API_ID);
		_api._data_compile_parameters = (DataCompileParametersApi *)get_engine_api(DATA_COMPILE_PARAMETERS_API_ID);
		_api._resource_manager = (ResourceManagerApi *)get_engine_api(RESOURCE_MANAGER_API_ID);
		_api._application = (ApplicationApi *)get_engine_api(APPLICATION_API_ID);
		_api._allocator = (AllocatorApi *)get_engine_api(ALLOCATOR_API_ID);
		_api._allocator_object = _api._allocator->make_plugin_allocator("TFPlugin");
		_tensorflow_allocator = SPF::ApiAllocator(_api._allocator, _api._allocator_object);
		_compiler_api_initialized = true;
	}

	void init_game_api(GetApiFunction get_engine_api)
	{
		_api._render_buffer = (RenderBufferApi*)get_engine_api(RENDER_BUFFER_API_ID);
		_api._render_interface = (RenderInterfaceApi*)get_engine_api(RENDER_INTERFACE_API_ID);
		_api._capture = (StreamCaptureApi*)get_engine_api(STREAM_CAPTURE_API_ID);
		_api._mesh = (MeshObjectApi*)get_engine_api(MESH_API_ID);
		_api._material = (MaterialApi*)get_engine_api(MATERIAL_API_ID);
		_api._lua = (LuaApi *)get_engine_api(LUA_API_ID);
		_api._logging = (LoggingApi*)get_engine_api(LOGGING_API_ID);
		_api._error = (ErrorApi*)get_engine_api(ERROR_API_ID);
		_api._file_system = (FileSystemApi *)get_engine_api(FILESYSTEM_API_ID);
		_api._resource_manager = (ResourceManagerApi *)get_engine_api(RESOURCE_MANAGER_API_ID);
		_api._options = (ApplicationOptionsApi*)get_engine_api(APPLICATION_OPTIONS_API_ID);
		_api._allocator = (AllocatorApi *)get_engine_api(ALLOCATOR_API_ID);
		_api._allocator_object = _api._allocator->make_plugin_allocator("TFPlugin");
		_tensorflow_allocator = SPF::ApiAllocator(_api._allocator, _api._allocator_object);
		_game_api_initialized = true;
	}

	void deinit_game_api()
	{
		_api._allocator->destroy_plugin_allocator(_api._allocator_object);
		_api._allocator = nullptr;
		_api._render_buffer = nullptr;
		_api._render_interface = nullptr;
		_api._capture = nullptr;
		_api._mesh = nullptr;
		_api._material = nullptr;
		_api._lua = nullptr;
		_api._error = nullptr;
		_api._logging = nullptr;
		_api._file_system = nullptr;
		_api._resource_manager = nullptr;
		_game_api_initialized = false;
	}

	void deinit_compiler_api()
	{
		_api._data_compiler = nullptr;
		_api._data_compile_parameters = nullptr;
		_api._allocator->destroy_plugin_allocator(_api._allocator_object);
		_api._data_compiler = nullptr;
		_api._data_compile_parameters = nullptr;
		_api._resource_manager = nullptr;
		_api._application = nullptr;
		_api._allocator = nullptr;
		_compiler_api_initialized = false;
	}

	bool TFPlugin::read_tf_graph(const std::string &path, unsigned mode, TF::GraphDef *def)
	{
		TF::Status status;
		if (mode == 0)
			status = TF::ReadBinaryProto(TF::Env::Default(), path, def);
		else if (mode == 1)
			status = TF::ReadTextProto(TF::Env::Default(), path, def);

		if (!status.ok())
		{
			_api._logging->error(get_name(), status.ToString().c_str());
			return false;
		}

		return true;
	}

	// Exposed to LUA
	void TFPlugin::run_tf_graph(const char *texture_name, const char *graph_name, const char *node, unsigned iterations)
	{
		session = MAKE_NEW(get_allocator(), Graph_Execution_Session);
		session->output_node_name = node;
		session->iterations_done = 0;
		session->iterations_max = iterations;
		RenderResource* resource_to_override = _api._render_interface->render_resource(texture_name, 0);
		ID3D11Texture2D* source = reinterpret_cast<ID3D11Texture2D*>(_api._render_interface->texture_2d(resource_to_override).texture);
		ID3D11Device* device = reinterpret_cast<ID3D11Device*>(_api._render_interface->device());
		ID3D11DeviceContext *immediate_context;
		device->GetImmediateContext(&immediate_context);
		RB_TextureBufferView view;
		D3D11_TEXTURE2D_DESC desc;
		source->GetDesc(&desc);
		view.format = _api._render_buffer->format(RB_INTEGER_COMPONENT, false, true, 8, 8, 8, 8);
		view.depth = 0;
		view.slices = 1;
		view.mip_levels = 1;
		view.width = session->texture_width = desc.Width;
		view.height = session->texture_height = desc.Height;
		
		view.type = RB_TEXTURE_TYPE_2D;

		RtlZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
		desc.Width = view.width;
		desc.Height = view.height;
		desc.MipLevels = view.mip_levels;
		desc.ArraySize = view.slices;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		device->CreateTexture2D(&desc, nullptr, &session->cuda_texture);
		cudaGraphicsD3D11RegisterResource(&session->cuda_resource, session->cuda_texture, cudaGraphicsRegisterFlagsNone);
		checkCUDAError("cudaGraphicsD3D11RegisterResource() failed");
		void* cudaLinearMemory = nullptr;
		size_t pitch = 0;
		cudaMallocPitch(&cudaLinearMemory, &pitch, session->texture_width * sizeof(unsigned char) * 4, session->texture_height);
		TFCuda::set_cuda_pitch(pitch);
		checkCUDAError("cudaMallocPitch() failed");
		cudaMemset(cudaLinearMemory, 1, pitch * session->texture_width);
		TFCuda::set_cuda_memory_pointer(cudaLinearMemory);

		auto buffer_size = session->texture_width * session->texture_height * _api._render_buffer->num_bits(view.format) / 8;
		session->override_texture_id = _api._render_buffer->create_buffer(buffer_size, RB_Validity::RB_VALIDITY_UPDATABLE, RB_View::RB_TEXTURE_BUFFER_VIEW, &view, nullptr);
		_api._render_buffer->resource_override(resource_to_override, _api._render_buffer->lookup_resource(session->override_texture_id));
		ID3D11Texture2D* destination = reinterpret_cast<ID3D11Texture2D*>(_api._render_interface->texture_2d(_api._render_buffer->lookup_resource(session->override_texture_id)).texture);

		// Triangle copy to verify that the cuda texture we compute on actually really has the image data
		immediate_context->CopyResource(session->cuda_texture, source);
		immediate_context->CopyResource(destination, session->cuda_texture);

		cudaGraphicsResourceSetMapFlags(session->cuda_resource, cudaGraphicsMapFlagsNone);
		cudaGraphicsMapResources(1, &session->cuda_resource);
		checkCUDAError("cudaGraphicsMapResources() failed");
		
		cudaGraphicsSubResourceGetMappedArray(&session->cuda_array, session->cuda_resource, 0, 0);
		checkCUDAError("cudaGraphicsSubResourceGetMappedArray() failed");

		// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
#ifdef MEASURE_TIME
		auto t1 = std::chrono::high_resolution_clock::now();
#endif
		cudaMemcpyFromArray(cudaLinearMemory, session->cuda_array, 0, 0, pitch * view.height, cudaMemcpyDeviceToDevice);
#ifdef MEASURE_TIME
		auto t2 = std::chrono::high_resolution_clock::now();
		auto time_amount = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
		_api._logging->warning(get_name(), _api._error->eprintf("Copy from 2D to linear memory took: `%lld` microseconds.", time_amount.count()));
#endif
		checkCUDAError("cudaMemcpyFromArray() failed");

		// Create tensor input data to fulfill graph conditions, could maybe refactored later
		session->zero_input = new TF::Tensor(TF::DT_FLOAT, TF::TensorShape({ 1, session->texture_width, session->texture_height, 4 }));

		// Create a new tensorflow session
		TF::SessionOptions options = TF::SessionOptions();
		options.config.mutable_gpu_options()->set_allow_growth(true);
		options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
		session->tf_session = TF::NewSession(options);

		// Load graph from file and add it to the session
		read_tf_graph(graph_name, 1, &session->tf_graph);
		TF::Status status = session->tf_session->Create(session->tf_graph);
		if (!status.ok()) {
			_api._logging->error(get_name(), status.ToString().c_str());
		}
	}

	void TFPlugin::end_tf_execution()
	{
		if (session)
		{
			cudaGraphicsUnmapResources(1, &session->cuda_resource);
			checkCUDAError("cudaGraphicsUnmapResources() failed");
			cudaGraphicsUnregisterResource(session->cuda_resource);
			checkCUDAError("cudaGraphicsUnregisterResource() failed");
			cudaFree(TFCuda::get_cuda_memory_pointer());
			checkCUDAError("cudaFree() failed");
			RenderResource *render_resource = _api._render_buffer->lookup_resource(session->override_texture_id);
			_api._render_buffer->release_resource_override(render_resource);
			session->tf_session->Close();
			MAKE_DELETE(get_allocator(), session);
			session = nullptr;
		}
	}

	void TFPlugin::setup_plugin(GetApiFunction get_engine_api)
	{	
		wait_for_debugger();

		if (!TF::IsGoogleCudaEnabled())
		{
			_api._logging->error(get_name(), "Could not initiate Tensorflow Plugin, no GPU support found.");
			return;
		}

		if (!_game_api_initialized)
			init_game_api(get_engine_api);
		setup_lua();

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
	}

	void TFPlugin::update_plugin(float dt)
	{
		if (session)
		{
			// Create input and output data structures and execute tensorflow graph
			std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = { { "input", *session->zero_input } };
			std::vector<TF::Tensor> out_tensors;

#ifdef MEASURE_TIME
			auto t1 = std::chrono::system_clock::now();
#endif
			TF::Status status = session->tf_session->Run({ inputs }, { session->output_node_name }, {}, &out_tensors);
#ifdef MEASURE_TIME
			auto t2 = std::chrono::system_clock::now();
			auto time_amount = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
			_api._logging->warning(get_name(), _api._error->eprintf("Running the Tensorflow Graph took: `%lld` milliseconds.", time_amount.count()));
#endif
			if (!status.ok()) {
				_api._logging->error(get_name(), status.ToString().c_str());
				end_tf_execution();
				return;
			}

#ifdef MEASURE_TIME
			auto t3 = std::chrono::high_resolution_clock::now();
#endif
			cudaMemcpy2DToArray(session->cuda_array, 0, 0, TFCuda::get_cuda_memory_pointer(), TFCuda::get_cuda_pitch(), session->texture_width * 4 * sizeof(unsigned char), session->texture_height, cudaMemcpyDeviceToDevice);
#ifdef MEASURE_TIME
			auto t4 = std::chrono::high_resolution_clock::now();
			auto time_amount2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
			_api._logging->warning(get_name(), _api._error->eprintf("Copy from linear to 2D memory took: `%lld` microseconds.", time_amount2.count()));
#endif
			checkCUDAError("cudaMemcpy2DToArray failed");

			ID3D11Texture2D* overrider = reinterpret_cast<ID3D11Texture2D*>(_api._render_interface->texture_2d(_api._render_buffer->lookup_resource(session->override_texture_id)).texture);
			ID3D11Device* device = reinterpret_cast<ID3D11Device*>(_api._render_interface->device());
			ID3D11DeviceContext *immediate_context;
			device->GetImmediateContext(&immediate_context);
			immediate_context->CopyResource(overrider, session->cuda_texture);
			++session->iterations_done;

			if (session->iterations_done >= session->iterations_max)
				end_tf_execution();
		}
	}

	bool TFPlugin::getLastCudaError(const char *errorMessage, const char *file, const int line)
	{
		cudaError_t err = cudaGetLastError();

		if (cudaSuccess != err)
		{
			_api._logging->error(get_name(), _api._error->eprintf("%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString(err)));
			return true;
		}
		return false;
	}

	void TFPlugin::shutdown_plugin()
	{
		end_tf_execution();
	}

	void TFPlugin::setup_data_compiler(GetApiFunction get_engine_api)
	{
		if (!_compiler_api_initialized)
			init_compiler_api(get_engine_api);
	}

	void TFPlugin::shutdown_data_compiler()
	{
		deinit_compiler_api();
	}

	const char *TFPlugin::get_name()
	{
		return "TensorflowPlugin";
	}

	int TFPlugin::can_refresh(uint64_t type)
	{
		return false;
	}
}
