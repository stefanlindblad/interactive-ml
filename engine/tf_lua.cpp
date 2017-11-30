#include "tf_lua.h"

namespace PLUGIN_NAMESPACE {

namespace {

	int run_graph(struct lua_State *L)
	{
		const char *texture_path = TFPlugin::get_api()._lua->tolstring(L, 1, nullptr);
		const char *graph_path = TFPlugin::get_api()._lua->tolstring(L, 2, nullptr);
		const char *node_name = TFPlugin::get_api()._lua->tolstring(L, 3, nullptr);
		unsigned iterations = (unsigned) TFPlugin::get_api()._lua->tointeger(L, 4);
		if (iterations == 0)
			iterations = 1;
		TFPlugin::run_tf_graph(texture_path, graph_path, node_name, iterations);
		return 0;
	}

} // anonymous namespace

void setup_lua()
{
	ApiInterface api = TFPlugin::get_api();
	api._lua->add_module_function("Tensorflow", "run_graph", run_graph);
}

} // PLUGIN_NAMESPACE
