#include "tf_lua.h"

namespace PLUGIN_NAMESPACE {

namespace {

	int run_graph(struct lua_State *L)
	{
		const char *graph_path = TFPlugin::get_api()._lua->tolstring(L, 1, nullptr);
		const char *node_name = TFPlugin::get_api()._lua->tolstring(L, 2, nullptr);
		unsigned iterations = (unsigned) TFPlugin::get_api()._lua->tointeger(L, 3);
		if (iterations == 0)
			iterations = 1;
		TFPlugin::run_tf_graph(graph_path, node_name, iterations);
		return 0;
	}

	int set_camera(struct lua_State *L)
	{
		CApiCamera* camera = (CApiCamera*) TFPlugin::get_api()._lua->topointer(L, 1);
		TFCuda::set_near_range(TFPlugin::get_api()._c->Camera->near_range(camera));
		TFCuda::set_far_range(TFPlugin::get_api()._c->Camera->far_range(camera));
		return 0;
	}

} // anonymous namespace

void setup_lua()
{
	ApiInterface api = TFPlugin::get_api();
	api._lua->add_module_function("Tensorflow", "run_graph", run_graph);
	api._lua->add_module_function("Tensorflow", "set_camera", set_camera);
}

} // PLUGIN_NAMESPACE
