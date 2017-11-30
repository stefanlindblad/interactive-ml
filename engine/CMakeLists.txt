cmake_minimum_required(VERSION 3.6)
project(tensorflow_plugin)

# Include common plugin CMake scripts and set the type of plugin to be to and ENGINE_PLUGIN
set(ENGINE_PLUGIN ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${REPOSITORY_DIR}/cmake")
include(CMakePlugin)

# Scan and add project source files
find_source_files(ALL_SOURCE_FILES)
find_cuda_files(ALL_CUDA_FILES)
find_package(CUDA)

# Add windows version resource if windows dll
if( PLATFORM_WINDOWS )
	if( BUILD_SHARED_LIBS )
		include_directories(${PROJECT_SOURCE_DIR})
		configure_file("${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.rc.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.rc")
		set(RESOURCE_FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.rc")
		source_group("Resources" FILES ${RESOURCE_FILES})
		list(APPEND ALL_SOURCE_FILES ${RESOURCE_FILES})
	endif()
endif()

# Define automatic namespace for C++
add_compile_options(-DPLUGIN_NAMESPACE=${PROJECT_NAME})

# Include plugin sdk files
include_directories(${REPOSITORY_DIR}/stingray_sdk)

# Necessary Tensorflow Includes & Links
include_directories(${TENSORFLOW_SRC_DIR})
include_directories(${TENSORFLOW_BUILD_DIR})
include_directories(${TENSORFLOW_BUILD_DIR}/external/eigen_archive)
include_directories(${TENSORFLOW_BUILD_DIR}//nsync/install/include)
include_directories(${TENSORFLOW_SRC_DIR}/third_party/eigen3)
include_directories(${TENSORFLOW_BUILD_DIR}/protobuf/src/protobuf/src)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${TENSORFLOW_BUILD_DIR}/Release)
link_directories(${TENSORFLOW_BUILD_DIR}/protobuf/src/protobuf/cmake/build/Release)

STRING(REGEX REPLACE "/W3" "/W0" ${PROJECT_NAME} ${CMAKE_CXX_FLAGS})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --disable-warnings")
set_source_files_properties(${ALL_CUDA_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Create target and set compile/link options
CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED
	${ALL_SOURCE_FILES}
)

# Necessary Tensorflow Linker Additions
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    libprotobuf
	tensorflow
	${CUDA_LIBRARIES}
)

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY LINK_FLAGS "/DEBUG /OPT:REF /OPT:ICF")
# Set target properties
set_system_properties(${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "${ENGINE_PLUGINS_FOLDER_NAME}")
set(TARGET_BASE_NAME "${PROJECT_NAME}_${ENGINE_PLUGIN_SUFFIX}_$<LOWER_CASE:$<CONFIG>>")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${TARGET_BASE_NAME}")
if( BUILD_SHARED_LIBS AND PLATFORM_IOS )
	set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "${ENGINE_IOS_CODE_SIGN_IDENTITY}")
endif()

# Set engine runtime plugin properties and enable hot-reloading.
set_plugin_runtime_output_directory("${TARGET_BASE_NAME}" "${ENGINE_PLUGINS_INSTALL_DIR}")
