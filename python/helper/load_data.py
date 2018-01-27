import OpenEXR
import Imath
import array
import numpy as np
from helper.constants import get_width, get_height, get_channels

IMG_WIDTH = get_width()
IMG_HEIGHT = get_height()
IMG_CHANNELS = get_channels()

def read_input_data(near, far, r_channel, g_channel, b_channel, d_channel):
    data = np.zeros([1,IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=np.float32)
    value_range = far - near
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            pos = y * IMG_WIDTH + x
            normalized_d = (d_channel[pos] - near) / value_range
            data[0, x, y, 0] = r_channel[pos]
            data[0, x, y, 1] = g_channel[pos]
            data[0, x, y, 2] = b_channel[pos]
            data[0, x, y, 3] = normalized_d
    return data

def read_truth_data(t_channel):
    data = np.zeros([1, IMG_WIDTH, IMG_HEIGHT, 1], dtype=np.float32)
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            pos = y * IMG_WIDTH + x
            data[0, x, y, 0] = t_channel[pos]
    return data


R,G,B,D = read_input_file("Input_pot_512.exr")
T = read_truth_file("AO_pot_512.exr")
    

def read_input_set(models, count, path):
    near_range = 0.1
    far_range = 1000.0
    models = []

    with open(modellocation) as f:
        for line in f:
            if line.endswith("\n"):
                line = line[:-1]
            models.append(line)

    model_count = 0
    data = np.zeros([len(models), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], dtype=np.float32)
    for model in models:
        input_filename = path + model + "/input_" + model + "_" + str(count) + ".exr"
        img_file = OpenEXR.InputFile(input_filename)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        R,G,B = [ array.array('f', img_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
        D = array.array('f', img_file.channel("depth.V", FLOAT)).tolist()
        value_range = far_range - near_range
        for y in range(IMG_HEIGHT):
            for x in range(IMG_WIDTH):
                pos = y * IMG_WIDTH + x
                normalized_d = (D[pos] - near_range) / value_range
                data[model_count, x, y, 0] = R[pos]
                data[model_count, x, y, 1] = G[pos]
                data[model_count, x, y, 2] = B[pos]
                data[model_count, x, y, 3] = normalized_d
        model_count += 1
    return data

def read_truth_set(models, count, path):
    models = []

    with open(modellocation) as f:
        for line in f:
            if line.endswith("\n"):
                line = line[:-1]
            models.append(line)

    model_count = 0
    data = np.zeros([len(models), IMG_WIDTH, IMG_HEIGHT, 1], dtype=np.float32)
    for model in models:
        truth_filename = path + model + "/groundtruth_" + model + "_" + str(count) + ".exr"
        img_file = OpenEXR.InputFile(truth_filename)
        T = array.array('f', img_file.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))).tolist()
        for y in range(IMG_HEIGHT):
            for x in range(IMG_WIDTH):
                pos = y * IMG_WIDTH + x
                data[model_count, x, y, 0] = T[pos]
        model_count += 1
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

def read_input_files(input_filenames):
    for filename in input_filenames:
        img_file = OpenEXR.InputFile(filename)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R,G,B = [ array.array('f', img_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
    D = array.array('f', img_file.channel("depth.V", FLOAT)).tolist()
    return R,G,B,D

def read_truth_files(truth_filenames):
    for filename in truth_filenames:
        img_file = OpenEXR.InputFile(filename)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    T = array.array('f', img_file.channel("R", FLOAT)).tolist()
    return T
