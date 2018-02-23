Welcome to the Autodesk Interactive ML Plugin
============================================================

This repository builds on top of the interactive default plugin (https://github.com/AutodeskGames/stingray-plugin).  
Check the Readme of the default plugin for common questions around installation and building the code.

[![Example Video of InteractiveML](https://img.youtube.com/vi/84CORMRR3ik/0.jpg)](https://www.youtube.com/watch?v=84CORMRR3ik)

## Dependencies

This Plugin links to the Tensorflow Library. You can either build it yourself:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/cmake

Or use the tensorflow.dll and libprotobuf.dll that are shipped with the plugin  
and are built for a x86-64 system architecture. These DLLs need to be  
reachable for your Autodesk Interactive Plugin DLL.

You need to set the following environment variables:
* TF_SRC_DIR => The location of the tensorflow source code
* TF_BUILD_DIR => The location of the tensorflow library

## Warranty
The whole code is provided "as is" and comes without any warranty or liability when being used.
