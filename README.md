Welcome to the Autodesk Interactive ML Plugin
============================================================

This repository builds on top of the interactive default plugin (https://github.com/AutodeskGames/stingray-plugin).
Check the Readme of this Plugin for common questions around installation and building the code.


## Dependencies

This Plugin links to the Tensorflow Library. You can either build it yourself:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/cmake

Or use the tensorflow.dll that ships with the plugin. The included library is build to
support GPU CUDA acceleration and AVX SIMD intruction extensions but without python bindings.

You need to set the following environment variables:
* TF_SRC_DIR => The location of the tensorflow source code
* TF_BUILD_DIR => The location of the tensorflow library


## WIP
This repository is currently work in progress. It is targeting to connect the interactive platform with 
Tensorflow to be able to use rendered frames to learn models and to apply pretrained TF models to the rendering.


## Current Work:
* General plugin architecture setup
* Making protobuf files available as resources in the engine