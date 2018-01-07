import OpenEXR
import Imath
import array
import numpy as np
from helper.constants import get_width, get_height, get_channels

IMG_WIDTH = get_width()
IMG_HEIGHT = get_height()
IMG_CHANNELS = get_channels()

def read_input_data(near, far, r_channel, g_channel, b_channel, d_channel):
    data = np.zeros([1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=np.float32)
    value_range = far - near
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            pos = y * IMG_WIDTH + x
            normalized_d = (d_channel[pos] - near) / value_range
            data[0, y, x, 0] = r_channel[pos]
            data[0, y, x, 1] = g_channel[pos]
            data[0, y, x, 2] = b_channel[pos]
            data[0, y, x, 3] = normalized_d
    return data

def read_truth_data(t_channel):
    data = np.zeros([1, IMG_HEIGHT, IMG_WIDTH, 1], dtype=np.float32)
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            pos = y * IMG_WIDTH + x
            data[0, y, x, 0] = t_channel[pos]
    return data

def read_input_file(input_filename):
    img_file = OpenEXR.InputFile(input_filename)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R,G,B = [ array.array('f', img_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
    D = array.array('f', img_file.channel("depth.V", FLOAT)).tolist()
    return R,G,B,D

def read_truth_file(truth_filename):
    img_file = OpenEXR.InputFile(truth_filename)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    T = array.array('f', img_file.channel("R", FLOAT)).tolist()
    return T
