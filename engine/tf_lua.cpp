#include "tf_lua.h"

namespace PLUGIN_NAMESPACE {

namespace {

	static bool nnao_preview = false;
	static bool nnao_multiply = false;

	int run_graph(struct lua_State *L)
	{
		bool endless = false;
		const char *graph_path = TFPlugin::get_api()._lua->tolstring(L, 1, nullptr);
		const char *node_name = TFPlugin::get_api()._lua->tolstring(L, 2, nullptr);
		unsigned iterations = (unsigned) TFPlugin::get_api()._lua->tointeger(L, 3);
		if (iterations == 0)
			endless = true;
		TFPlugin::run_tf_graph(graph_path, node_name, iterations, endless);
		return 0;
	}

	int set_camera(struct lua_State *L)
	{
		CApiCamera* camera = (CApiCamera*) TFPlugin::get_api()._lua->topointer(L, 1);
		TFCuda::set_near_range(TFPlugin::get_api()._c->Camera->near_range(camera));
		TFCuda::set_far_range(TFPlugin::get_api()._c->Camera->far_range(camera));
		return 0;
	}

	int toogle_nnao_preview(struct lua_State *L)
	{
		nnao_preview = !nnao_preview;
		static SPF::ConstConfigRoot setting = { SPF::const_config::BOOL, nnao_preview };
		TFPlugin::get_api()._render_interface->set_render_setting("nnao_map_visualization", &setting);
		return 0;
	}

	int toogle_nnao_multiply(struct lua_State *L)
	{
		nnao_multiply = !nnao_multiply;
		static SPF::ConstConfigRoot setting = { SPF::const_config::BOOL, nnao_multiply };
		TFPlugin::get_api()._render_interface->set_render_setting("nnao_scene_combine", &setting);
		return 0;
	}

} // anonymous namespace

void setup_lua()
{
	ApiInterface api = TFPlugin::get_api();
	api._lua->add_module_function("Tensorflow", "run_graph", run_graph);
	api._lua->add_module_function("Tensorflow", "set_camera", set_camera);
	api._lua->add_module_function("Tensorflow", "toogle_nnao_preview", toogle_nnao_preview);
	api._lua->add_module_function("Tensorflow", "toogle_nnao_multiply", toogle_nnao_multiply);
}

} // PLUGIN_NAMESPACE
