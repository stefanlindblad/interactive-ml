#include <engine_plugin_api/plugin_api.h>
#include <plugin_foundation/platform.h>
#include "tf_plugin.h"

extern "C"
{
	#ifdef STATIC_LINKING
		void *get_tensorflow_plugin_api(unsigned api)
		{
			if (api == PLUGIN_API_ID) {
				static struct PluginApi api = {0};
				api.setup_game = &PLUGIN_NAMESPACE::TFPlugin::setup_plugin;
				api.update_game = &PLUGIN_NAMESPACE::TFPlugin::update_plugin;
				api.shutdown_game = &PLUGIN_NAMESPACE::TFPlugin::shutdown_plugin;
				api.setup_data_compiler = &PLUGIN_NAMESPACE::TFPlugin::setup_data_compiler;
				api.shutdown_data_compiler = &PLUGIN_NAMESPACE::TFPlugin::shutdown_plugin;
				return &api;
			}
			return 0;
		}
	#else
		PLUGIN_DLLEXPORT void *get_plugin_api(unsigned api)
		{
			if (api == PLUGIN_API_ID) {
				static struct PluginApi api = {0};
				api.setup_game = &PLUGIN_NAMESPACE::TFPlugin::setup_plugin;
				api.update_game = &PLUGIN_NAMESPACE::TFPlugin::update_plugin;
				api.shutdown_game = &PLUGIN_NAMESPACE::TFPlugin::shutdown_plugin;
				api.setup_data_compiler = &PLUGIN_NAMESPACE::TFPlugin::setup_data_compiler;
				api.shutdown_data_compiler = &PLUGIN_NAMESPACE::TFPlugin::shutdown_data_compiler;
				api.can_refresh = &PLUGIN_NAMESPACE::TFPlugin::can_refresh;
				return &api;
			}
			return 0;
		}
	#endif
}
