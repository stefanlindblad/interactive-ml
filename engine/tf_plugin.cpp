#include "tf_plugin.h"

namespace PLUGIN_NAMESPACE
{
	//#define WAITFORDEBUGGER
	#define checkCUDAError(msg) if(getLastCudaError (msg, __FILE__, __LINE__)) return
	//#define MEASURE_TIME
	//#define PRINT_RESULTS

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

	// This is actually bad, should find a more flexible solution at some point
	const unsigned NUMBER_OF_CHANNELS = 4;

	// pointers to the render resources we are flushing through the network
	enum RenderTargetStep { ReceivingNormals, ReceivingDepth, ReceivingNNAO, ReceivedEverything };
	static RenderResource *normals_resource = nullptr;
	static ID3D11Texture2D *normals_render_target = nullptr;
	static RenderResource *depth_resource = nullptr;
	static ID3D11Texture2D *depth_render_target = nullptr;
	static RenderResource *nnao_resource = nullptr;
	static ID3D11Texture2D *nnao_render_target = nullptr;
	static RenderTargetStep step_identifier = ReceivingNormals;

	// Structure to define a tensorflow session, only one is supported right now
	struct Graph_Execution_Session
	{
		bool initialized = false;
		bool endless = false;
		unsigned texture_width;
		unsigned texture_height;
		unsigned iterations_done;
		unsigned iterations_max;
		std::string output_node_name;
		std::string tf_graph_name;
		cudaArray *input_array = nullptr;
		cudaArray *depth_array = nullptr;
		cudaArray *output_array = nullptr;
		cudaGraphicsResource *input_resource = nullptr;
		ID3D11Texture2D *input_texture = nullptr;
		cudaGraphicsResource *depth_resource = nullptr;
		ID3D11Texture2D *depth_texture = nullptr;
		cudaGraphicsResource *output_resource = nullptr;
		ID3D11Texture2D *output_texture = nullptr;
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
		_api._data_compiler = static_cast<DataCompilerApi *>(get_engine_api(DATA_COMPILER_API_ID));
		_api._data_compile_parameters = static_cast<DataCompileParametersApi*>(get_engine_api(DATA_COMPILE_PARAMETERS_API_ID));
		_api._resource_manager = static_cast<ResourceManagerApi*>(get_engine_api(RESOURCE_MANAGER_API_ID));
		_api._application = static_cast<ApplicationApi*>(get_engine_api(APPLICATION_API_ID));
		_api._allocator = static_cast<AllocatorApi*>(get_engine_api(ALLOCATOR_API_ID));
		_api._allocator_object = _api._allocator->make_plugin_allocator(TFPlugin::get_name());
		_tensorflow_allocator = SPF::ApiAllocator(_api._allocator, _api._allocator_object);
		_compiler_api_initialized = true;
	}

	void init_game_api(GetApiFunction get_engine_api)
	{
		_api._render_buffer = static_cast<RenderBufferApi*>(get_engine_api(RENDER_BUFFER_API_ID));
		_api._render_interface = static_cast<RenderInterfaceApi*>(get_engine_api(RENDER_INTERFACE_API_ID));
		_api._capture = static_cast<StreamCaptureApi*>(get_engine_api(STREAM_CAPTURE_API_ID));
		_api._mesh = static_cast<MeshObjectApi*>(get_engine_api(MESH_API_ID));
		_api._material = static_cast<MaterialApi*>(get_engine_api(MATERIAL_API_ID));
		_api._lua = static_cast<LuaApi*>(get_engine_api(LUA_API_ID));
		_api._logging = static_cast<LoggingApi*>(get_engine_api(LOGGING_API_ID));
		_api._error = static_cast<ErrorApi*>(get_engine_api(ERROR_API_ID));
		_api._file_system = static_cast<FileSystemApi*>(get_engine_api(FILESYSTEM_API_ID));
		_api._resource_manager = static_cast<ResourceManagerApi*>(get_engine_api(RESOURCE_MANAGER_API_ID));
		_api._options = static_cast<ApplicationOptionsApi*>(get_engine_api(APPLICATION_OPTIONS_API_ID));
		_api._c = static_cast<CApi*>(get_engine_api(C_API_ID));
		_api._allocator = static_cast<AllocatorApi*>(get_engine_api(ALLOCATOR_API_ID));
		_api._allocator_object = _api._allocator->make_plugin_allocator(TFPlugin::get_name());
		_tensorflow_allocator = SPF::ApiAllocator(_api._allocator, _api._allocator_object);
		_game_api_initialized = true;
	}

	void deinit_game_api()
	{
		_api._allocator->destroy_plugin_allocator(_api._allocator_object);
		_api._allocator_object = nullptr;
		_api._allocator = nullptr;
		_api._render_buffer = nullptr;
		_api._render_interface = nullptr;
		_api._capture = nullptr;
		_api._mesh = nullptr;
		_api._material = nullptr;
		_api._lua = nullptr;
		_api._logging = nullptr;
		_api._error = nullptr;
		_api._file_system = nullptr;
		_api._resource_manager = nullptr;
		_api._options = nullptr;
		_api._c = nullptr;
		_game_api_initialized = false;
	}

	void deinit_compiler_api()
	{
		_api._allocator->destroy_plugin_allocator(_api._allocator_object);
		_api._allocator_object = nullptr;
		_api._allocator = nullptr;
		_api._data_compiler = nullptr;
		_api._data_compile_parameters = nullptr;
		_api._resource_manager = nullptr;
		_api._application = nullptr;
		_compiler_api_initialized = false;
	}

	TF::Status TFPlugin::read_tf_graph(const std::string &path, unsigned mode, TF::GraphDef *def)
	{
		TF::Status status;
		if (mode == 0)
			status = TF::ReadBinaryProto(TF::Env::Default(), path, def);
		else if (mode == 1)
			status = TF::ReadTextProto(TF::Env::Default(), path, def);

		return status;
	}

	// Exposed to LUA
	void TFPlugin::run_tf_graph(const char *graph_name, const char *node, unsigned iterations, bool endless)
	{
		if (normals_render_target == nullptr || depth_render_target == nullptr || nnao_render_target == nullptr)
		{
			_api._logging->error(get_name(), "Could not initialize Tensorflow Graph Session, Render Targets are missing.");
			return;
		}

		session = MAKE_NEW(get_allocator(), Graph_Execution_Session);
		session->output_node_name = node;
		session->tf_graph_name = graph_name;
		session->iterations_done = 0;
		session->iterations_max = iterations;
		session->endless = endless;

		ID3D11Device* device = reinterpret_cast<ID3D11Device*>(_api._render_interface->device());
		ID3D11DeviceContext *immediate_context;
		device->GetImmediateContext(&immediate_context);

		D3D11_TEXTURE2D_DESC desc;
		normals_render_target->GetDesc(&desc);
		session->texture_width = desc.Width;
		session->texture_height = desc.Height;
		RtlZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
		desc.Width = session->texture_width;
		desc.Height = session->texture_height;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

		desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
		device->CreateTexture2D(&desc, nullptr, &session->input_texture);

		desc.Format = DXGI_FORMAT_R32_FLOAT;
		device->CreateTexture2D(&desc, nullptr, &session->depth_texture);
		device->CreateTexture2D(&desc, nullptr, &session->output_texture);

		cudaGraphicsD3D11RegisterResource(&session->input_resource, session->input_texture, cudaGraphicsRegisterFlagsNone);
		checkCUDAError("cudaGraphicsD3D11RegisterResource() failed");
		cudaGraphicsD3D11RegisterResource(&session->depth_resource, session->depth_texture, cudaGraphicsRegisterFlagsNone);
		checkCUDAError("cudaGraphicsD3D11RegisterResource() failed");
		cudaGraphicsD3D11RegisterResource(&session->output_resource, session->output_texture, cudaGraphicsRegisterFlagsNone);
		checkCUDAError("cudaGraphicsD3D11RegisterResource() failed");

		size_t memorySize = session->texture_width * sizeof(unsigned char) * NUMBER_OF_CHANNELS;
		void* inputLinearMemory = nullptr;
		void* depthLinearMemory = nullptr;
		void* outputLinearMemory = nullptr;
		size_t pitchSize = 0;

		cudaMallocPitch(&inputLinearMemory, &pitchSize, memorySize, session->texture_height);
		checkCUDAError("cudaMallocPitch() failed");
		cudaMallocPitch(&depthLinearMemory, &pitchSize, memorySize, session->texture_height);
		checkCUDAError("cudaMallocPitch() failed");
		cudaMallocPitch(&outputLinearMemory, &pitchSize, memorySize, session->texture_height);
		checkCUDAError("cudaMallocPitch() failed");

		cudaMemset(inputLinearMemory, 0, pitchSize * session->texture_height);
		checkCUDAError("cudaMemset() failed");
		cudaMemset(depthLinearMemory, 0, pitchSize * session->texture_height);
		checkCUDAError("cudaMemset() failed");
		cudaMemset(outputLinearMemory, 0, pitchSize * session->texture_height);
		checkCUDAError("cudaMemset() failed");

		cudaGraphicsResourceSetMapFlags(session->input_resource, cudaGraphicsMapFlagsNone);
		checkCUDAError("cudaGraphicsResourceSetMapFlags() failed");
		cudaGraphicsResourceSetMapFlags(session->depth_resource, cudaGraphicsMapFlagsNone);
		checkCUDAError("cudaGraphicsResourceSetMapFlags() failed");
		cudaGraphicsResourceSetMapFlags(session->output_resource, cudaGraphicsMapFlagsNone);

		checkCUDAError("cudaGraphicsResourceSetMapFlags() failed");
		cudaGraphicsMapResources(1, &session->input_resource);
		checkCUDAError("cudaGraphicsMapResources() failed");
		cudaGraphicsMapResources(1, &session->depth_resource);
		checkCUDAError("cudaGraphicsMapResources() failed");
		cudaGraphicsMapResources(1, &session->output_resource);
		checkCUDAError("cudaGraphicsMapResources() failed");

		cudaGraphicsSubResourceGetMappedArray(&session->input_array, session->input_resource, 0, 0);
		checkCUDAError("cudaGraphicsSubResourceGetMappedArray() failed");
		cudaGraphicsSubResourceGetMappedArray(&session->depth_array, session->depth_resource, 0, 0);
		checkCUDAError("cudaGraphicsSubResourceGetMappedArray() failed");
		cudaGraphicsSubResourceGetMappedArray(&session->output_array, session->output_resource, 0, 0);
		checkCUDAError("cudaGraphicsSubResourceGetMappedArray() failed");

		TFCuda::set_input_memory_pointer(inputLinearMemory);
		TFCuda::set_depth_memory_pointer(depthLinearMemory);
		TFCuda::set_output_memory_pointer(outputLinearMemory);
		TFCuda::set_pitch(pitchSize);
		
		// Create tensor input data to fulfill graph conditions, could maybe refactored later
		session->zero_input = new TF::Tensor(TF::DT_FLOAT, TF::TensorShape({ 1, session->texture_width, session->texture_height, 4 }));

		// Create a new Tensorflow Session
		TF::SessionOptions options = TF::SessionOptions();
		options.config.mutable_gpu_options()->set_allow_growth(true);

		session->tf_session = TF::NewSession(options);
		TF::Status status = read_tf_graph(session->tf_graph_name, 0, &session->tf_graph);
		if (!status.ok()) {
			_api._logging->error(get_name(), status.ToString().c_str());
			return;
		}

		status = session->tf_session->Create(session->tf_graph);
		if (!status.ok()) {
			_api._logging->error(get_name(), status.ToString().c_str());
			return;
		}

		session->initialized = true;
	}

	void TFPlugin::end_tf_execution()
	{
		if (session)
		{
			cudaGraphicsUnmapResources(1, &session->depth_resource);
			checkCUDAError("cudaGraphicsUnmapResources() failed");
			cudaGraphicsUnregisterResource(session->depth_resource);
			checkCUDAError("cudaGraphicsUnregisterResource() failed");
			cudaFree(TFCuda::get_depth_memory_pointer());
			checkCUDAError("cudaFree() failed");
			session->depth_texture->Release();

			cudaGraphicsUnmapResources(1, &session->input_resource);
			checkCUDAError("cudaGraphicsUnmapResources() failed");
			cudaGraphicsUnregisterResource(session->input_resource);
			checkCUDAError("cudaGraphicsUnregisterResource() failed");
			cudaFree(TFCuda::get_input_memory_pointer());
			checkCUDAError("cudaFree() failed");
			session->input_texture->Release();

			cudaGraphicsUnmapResources(1, &session->output_resource);
			checkCUDAError("cudaGraphicsUnmapResources() failed");
			cudaGraphicsUnregisterResource(session->output_resource);
			checkCUDAError("cudaGraphicsUnregisterResource() failed");
			cudaFree(TFCuda::get_output_memory_pointer());
			checkCUDAError("cudaFree() failed");
			session->output_texture->Release();

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
		setup_kernels();
	}

	void TFPlugin::update_plugin(float dt)
	{}

	void TFPlugin::end_frame()
	{
		step_identifier = ReceivingNormals;
	}

	void TFPlugin::render(RenderDevicePluginArguments *arguments)
	{
		RenderResource *target = static_cast<RenderResource*>(arguments->engine_data.render_target);

		switch (step_identifier) {
			case ReceivingNormals:
				normals_resource = target;
				normals_render_target = reinterpret_cast<ID3D11Texture2D*>(_api._render_interface->texture_2d(normals_resource).texture);
				step_identifier = ReceivingDepth;
				break;

			case ReceivingDepth:
				depth_resource = target;
				depth_render_target = reinterpret_cast<ID3D11Texture2D*>(_api._render_interface->texture_2d(depth_resource).texture);
				step_identifier = ReceivingNNAO;
				break;

			case ReceivingNNAO:
				nnao_resource = target;
				nnao_render_target = reinterpret_cast<ID3D11Texture2D*>(_api._render_interface->texture_2d(nnao_resource).texture);
				step_identifier = ReceivedEverything;
				break;

			default:
				return;
		}

		if (step_identifier == ReceivedEverything && session && session->initialized)
		{
			ID3D11Device* device = reinterpret_cast<ID3D11Device*>(_api._render_interface->device());
			ID3D11DeviceContext *immediate_context;
			device->GetImmediateContext(&immediate_context);

			cudaMemset(TFCuda::get_input_memory_pointer(), 0, TFCuda::get_pitch() * session->texture_height);
			checkCUDAError("cudaMemset() failed");
			cudaMemset(TFCuda::get_depth_memory_pointer(), 0, TFCuda::get_pitch() * session->texture_height);
			checkCUDAError("cudaMemset() failed");
			cudaMemset(TFCuda::get_output_memory_pointer(), 0, TFCuda::get_pitch() * session->texture_height);
			checkCUDAError("cudaMemset() failed");

			// Copy the normals texture data (R8G8B8A8) into CUDA memory
			immediate_context->CopySubresourceRegion(session->input_texture, 0, 0, 0, 0, normals_render_target, 0, nullptr);
			cudaMemcpy2DFromArray(TFCuda::get_input_memory_pointer(), TFCuda::get_pitch(), session->input_array, 0, 0, session->texture_width * sizeof(unsigned char) * NUMBER_OF_CHANNELS, session->texture_height, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy2DFromArray() failed");

			// Copy the depth texture data (R32F) into CUDA memory
			immediate_context->CopySubresourceRegion(session->depth_texture, 0, 0, 0, 0, depth_render_target, 0, nullptr);
			cudaMemcpy2DFromArray(TFCuda::get_depth_memory_pointer(), TFCuda::get_pitch(), session->depth_array, 0, 0, session->texture_width * sizeof(float), session->texture_height, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy2DFromArray() failed");

			std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = { { "image_data", *session->zero_input } };
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

#ifdef PRINT_RESULTS
			// Prints the first 500 tensor values, this requires the output operator to run on the host
			if (session->iterations_done == 1)
			{
				TF::Tensor result = out_tensors.at(0);
				auto output_flat = result.flat<float>();

				for (auto i = 0; i < 500; i++) {
					float value = output_flat(i);
					_api._logging->info(get_name(), _api._error->eprintf("Tensor Value at Position %i is: %f", i, value));
				}
			}
#endif
			cudaMemcpy2DToArray(session->output_array, 0, 0, TFCuda::get_output_memory_pointer(), TFCuda::get_pitch(), session->texture_width * sizeof(float), session->texture_height, cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy2DToArray failed");

			cudaDeviceSynchronize();
			checkCUDAError("cudaDeviceSynchronize failed");

			immediate_context->CopySubresourceRegion(nnao_render_target, 0, 0, 0, 0, session->output_texture, 0, nullptr);
			++session->iterations_done;
			
			if (!session->endless && session->iterations_done >= session->iterations_max)
				end_tf_execution();

			step_identifier = ReceivingNormals;
		}
	}

	bool TFPlugin::getLastCudaError(const char *errorMessage, const char *file, const int line)
	{
		cudaError_t err = cudaGetLastError();

		if (cudaSuccess != err)
		{
			_api._logging->error(get_name(), _api._error->eprintf("%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, static_cast<int>(err), cudaGetErrorString(err)));

			if (session)
			{
				cudaGraphicsUnmapResources(1, &session->depth_resource);
				cudaGraphicsUnregisterResource(session->depth_resource);
				cudaFree(TFCuda::get_depth_memory_pointer());
				session->depth_texture->Release();

				cudaGraphicsUnmapResources(1, &session->input_resource);
				cudaGraphicsUnregisterResource(session->input_resource);
				cudaFree(TFCuda::get_input_memory_pointer());
				session->input_texture->Release();

				cudaGraphicsUnmapResources(1, &session->output_resource);
				cudaGraphicsUnregisterResource(session->output_resource);
				cudaFree(TFCuda::get_output_memory_pointer());
				session->output_texture->Release();

				MAKE_DELETE(get_allocator(), session);
				session = nullptr;
			}

			return true;
		}
		return false;
	}

	void TFPlugin::shutdown_plugin()
	{
		end_tf_execution();
		deinit_game_api();
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

	void* TFPlugin::get_render_env()
	{
		return nullptr;
	}
}
