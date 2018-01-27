import OpenEXR
import Imath
import numpy as np
import array
import tensorflow as tf
from tensorflow.python.tools import freeze_graph as freezer
from helper.constants import get_width, get_height

IMG_WIDTH = get_width()
IMG_HEIGHT = get_height()

def freeze_graph(graph_structure_path, variable_checkpoint_path, frozen_graph_name):
    interactive_module = tf.load_op_library("C:/code/tensorflow/tensorflow/contrib/cmake/build/Release/interactive_ops.dll")
    freezer.freeze_graph(graph_structure_path, "",
                          False, variable_checkpoint_path, "InteractiveOutput",
                          "save/restore_all", "save/Const:0",
                          frozen_graph_name, True, "")

def write_output_file(filename, data):
    width = data.shape[0]
    height = data.shape[1]
    exr = OpenEXR.OutputFile(filename, OpenEXR.Header(width, height))
    output = array.array('f', [ 0.0 ] * (height * width))
    for y in range(height):
        for x in range(width):
            output[y * width + x] = data[x, y]
    output = output.tostring()
    exr.writePixels({'R': output, 'G': output, 'B': output})
    exr.close()

def write_output_array(filename, data):
    height = IMG_HEIGHT
    width = IMG_WIDTH
    struct = np.zeros([width, height], dtype=np.float32)
    for y in range(height):
        for x in range(width):
            struct[x, y] = data[y * width + x]
    exr = OpenEXR.OutputFile(filename, OpenEXR.Header(width, height))
    output = array.array('f', [ 0.0 ] * (height * width))
    for y in range(height):
        for x in range(width):
            output[y * width + x] = struct[x, y]
    output = output.tostring()
    exr.writePixels({'R': output, 'G': output, 'B': output})
    exr.close()

def write_output_array2(filename, r, g, b, d):
    height = IMG_HEIGHT
    width = IMG_WIDTH
    struct = np.zeros([width, height, 4], dtype=np.float32)
    for y in range(height):
        for x in range(width):
            struct[x, y, 0] = r[y * width + x]
            struct[x, y, 1] = g[y * width + x]
            struct[x, y, 2] = b[y * width + x]
            struct[x, y, 3] = d[y * width + x]
    header = OpenEXR.Header(width, height)
    header['channels'] = {'R' : Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)), 'G' : Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
    'B' : Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)), 'depth.V' : Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)), 'A' : Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))}
    exr = OpenEXR.OutputFile(filename, header)
    output1 = array.array('f', [ 0.0 ] * (height * width))
    output2 = array.array('f', [ 0.0 ] * (height * width))
    output3 = array.array('f', [ 0.0 ] * (height * width))
    output4 = array.array('f', [ 0.0 ] * (height * width))
    for y in range(height):
        for x in range(width):
            output1[y * width + x] = struct[x, y, 0]
            output2[y * width + x] = struct[x, y, 1]
            output3[y * width + x] = struct[x, y, 2]
            output4[y * width + x] = struct[x, y, 3]
    output1 = output1.tostring()
    output2 = output2.tostring()
    output3 = output3.tostring()
    output4 = output4.tostring()
    exr.writePixels({'R': output1, 'G': output2, 'B': output3, 'depth.V': output4})
    exr.close()

