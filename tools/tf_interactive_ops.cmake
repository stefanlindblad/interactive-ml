# Instructions how to build a the custom tensorflow ops for the interactive engine

# 1. save this file as tensorflow/contrib/cmake/tf_interactive_ops.cmake
# 2. append "include(tf_interactive_ops.cmake)" to CMakeLists.txt in the same folder at the end of the file
# 3. copy the kernels subfolder in the repository to tensorflow/contrib/cmake/kernels
# 3. call cmake (see tensorflow/contrib/cmake/README.md)
# 4. you need to build the source tree once to generate all header files needed, ie:
#     MSBuild /p:Configuration=Release tf_python_build_pip_package.vcxproj
# 5. build the interactive_ops
#     MSBuild /p:Configuration=Release interactive_ops.vcxproj
#    or if you know what you are doing you can skip checking dependencies:
#     MSBuild /p:Configuration=Release /p:BuildProjectReferences=false interactive_ops.vcxproj

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  --whole-archive")

set(tf_interactive_srcs
    "kernels/tf_kernel.cpp"
    "kernels/tf_kernel.h"
)
set(tf_interactive_gpu_srcs
	"kernels/tf_kernel.cu.cc"
)

set(target_dir
    "${CMAKE_INSTALL_PREFIX}/interactive_ops"
)

AddUserOps(TARGET interactive_ops
    SOURCES ${tf_interactive_srcs}
    GPUSOURCES ${tf_interactive_gpu_srcs}
    DEPENDS pywrap_tensorflow_internal tf_python_ops
)

target_link_libraries(interactive_ops -Wl,--whole-archive ${foo_location} -Wl,--no-whole-archive)