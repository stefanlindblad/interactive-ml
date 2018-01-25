import tensorflow as tf
from helper.constants import get_width, get_height, get_channels

IMG_WIDTH = get_width()
IMG_HEIGHT = get_height()
IMG_CHANNELS = get_channels()

CONV_FILTER_SIZE = 3
DECONV_FILTER_SIZE = 4
CONV_STRIDES = [1, 1, 1, 1]
POOL_STRIDES = [1, 2, 2, 1]
DECONV_STRIDES = [1, 2, 2, 1]
POOL_CONFIG = [1, 2, 2, 1]
CONV_0_SIZE = 8
CONV_1_SIZE = 16
CONV_2_SIZE = 32
CONV_3_SIZE = 64
CONV_4_SIZE = 128
DEFAULT_DEV = 0.01
PADDING_CONFIG = "SAME"
DATA_FORMAT_CONFIG = "NHWC"

# Building NNAO Network Structure
def build_nnao_network(print_shapes=False):
    image_data = tf.placeholder(dtype=tf.float32, shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name="image_data")
    ground_truth = tf.placeholder(dtype=tf.float32, shape=[None, IMG_WIDTH, IMG_HEIGHT, 1], name="ground_truth")
    batch_size = tf.shape(image_data)[0]
    if print_shapes:
        print("Image Data: ", image_data.shape)

    # Level 0 Down
    weight_conv_0d = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, IMG_CHANNELS, CONV_0_SIZE], stddev=DEFAULT_DEV))
    bias_conv_0d = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_0_SIZE]))
    transposed_data = tf.transpose(image_data, [0, 2, 1, 3], name="start_transpose")
    conv_0d = tf.nn.conv2d(input=transposed_data, filter=weight_conv_0d, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_0_down")
    relu_0d = tf.nn.relu(features=conv_0d + bias_conv_0d, name="relu_0_down")
    pooling_0d = tf.nn.avg_pool(value=relu_0d, ksize=POOL_CONFIG, strides=POOL_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="pooling_0_down")
    if print_shapes:
        print("0 Down: ", pooling_0d.shape)

    # Level 1 Down
    weight_conv_1d = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_0_SIZE, CONV_1_SIZE], stddev=DEFAULT_DEV))
    bias_conv_1d = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_1_SIZE]))
    conv_1d = tf.nn.conv2d(input=pooling_0d, filter=weight_conv_1d, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_1_down")
    relu_1d = tf.nn.relu(features=conv_1d + bias_conv_1d, name="relu_1_down")
    pooling_1d = tf.nn.avg_pool(value=relu_1d, ksize=POOL_CONFIG, strides=POOL_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="pooling_1_down")
    if print_shapes:
        print("1 Down: ", pooling_1d.shape)

    # Level 2 Down
    weight_conv_2d = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_1_SIZE, CONV_2_SIZE], stddev=DEFAULT_DEV))
    bias_conv_2d = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_2_SIZE]))
    conv_2d = tf.nn.conv2d(input=pooling_1d, filter=weight_conv_2d, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_2_down")
    relu_2d = tf.nn.relu(features=conv_2d + bias_conv_2d, name="relu_2_down")
    pooling_2d = tf.nn.avg_pool(value=relu_2d, ksize=POOL_CONFIG, strides=POOL_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="pooling_2_down")
    if print_shapes:
        print("2 Down: ", pooling_2d.shape)

    # Level 3 Down
    weight_conv_3d = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_2_SIZE, CONV_3_SIZE], stddev=DEFAULT_DEV))
    bias_conv_3d = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_3_SIZE]))
    conv_3d = tf.nn.conv2d(input=pooling_2d, filter=weight_conv_3d, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_3_down")
    relu_3d = tf.nn.relu(features=conv_3d + bias_conv_3d, name="relu_3_down")
    pooling_3d = tf.nn.avg_pool(value=relu_3d, ksize=POOL_CONFIG, strides=POOL_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="pooling_3_down")
    if print_shapes:
        print("3 Down: ", pooling_3d.shape)

    # Level 4 Bottom
    weight_conv_4b = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_3_SIZE, CONV_4_SIZE], stddev=DEFAULT_DEV))
    bias_conv_4b = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_4_SIZE]))
    conv_4b = tf.nn.conv2d(input=pooling_3d, filter=weight_conv_4b, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_4_down")
    relu_4b = tf.nn.relu(features=conv_4b + bias_conv_4b, name="relu_4_down")
    if print_shapes:
        print("4 Bottom: ", relu_4b.shape)

    # Level 3 Up
    weight_conv_3u = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_4_SIZE, CONV_3_SIZE], stddev=DEFAULT_DEV))
    weight_deconv_3u = tf.Variable(tf.truncated_normal([DECONV_FILTER_SIZE, DECONV_FILTER_SIZE, CONV_3_SIZE, CONV_4_SIZE], stddev=DEFAULT_DEV))
    bias_conv_3u = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_3_SIZE]))
    deconv_3u_shape = tf.stack([batch_size, int(relu_4b.shape[1]*2), int(relu_4b.shape[2]*2), CONV_3_SIZE])
    deconv_3u = tf.nn.conv2d_transpose(value=relu_4b, filter=weight_deconv_3u, output_shape=deconv_3u_shape, strides=DECONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="deconvolution_3_up")
    concat_3u = tf.concat(values=[deconv_3u, relu_3d], axis=3, name="concat_3_up")
    conv_3u = tf.nn.conv2d(input=concat_3u, filter=weight_conv_3u, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_3_up")
    relu_3u = tf.nn.relu(features=conv_3u + bias_conv_3u, name="relu_3_up")
    if print_shapes:
        print("3 Up: ", relu_3u.shape)

    # Level 2 Up
    weight_conv_2u = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_3_SIZE, CONV_2_SIZE], stddev=DEFAULT_DEV))
    weight_deconv_2u = tf.Variable(tf.truncated_normal([DECONV_FILTER_SIZE, DECONV_FILTER_SIZE, CONV_2_SIZE, CONV_3_SIZE], stddev=DEFAULT_DEV))
    bias_conv_2u = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_2_SIZE]))
    deconv_2u_shape = tf.stack([batch_size, int(relu_3u.shape[1]*2), int(relu_3u.shape[2]*2), CONV_2_SIZE])
    deconv_2u = tf.nn.conv2d_transpose(value=relu_3u, filter=weight_deconv_2u, output_shape=deconv_2u_shape, strides=DECONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="deconvolution_2_up")
    concat_2u = tf.concat(values=[deconv_2u, relu_2d], axis=3, name="concat_2_up")
    conv_2u = tf.nn.conv2d(input=concat_2u, filter=weight_conv_2u, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_2_up")
    relu_2u = tf.nn.relu(features=conv_2u + bias_conv_2u, name="relu_2_up")
    if print_shapes:
        print("2 Up: ", relu_2u.shape)

    # Level 1 Up
    weight_conv_1u = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_2_SIZE, CONV_1_SIZE], stddev=DEFAULT_DEV))
    weight_deconv_1u = tf.Variable(tf.truncated_normal([DECONV_FILTER_SIZE, DECONV_FILTER_SIZE, CONV_1_SIZE, CONV_2_SIZE], stddev=DEFAULT_DEV))
    bias_conv_1u = tf.Variable(tf.constant(DEFAULT_DEV, shape=[CONV_1_SIZE]))
    deconv_1u_shape = tf.stack([batch_size, int(relu_2u.shape[1]*2), int(relu_2u.shape[2]*2), CONV_1_SIZE])
    deconv_1u = tf.nn.conv2d_transpose(value=relu_2u, filter=weight_deconv_1u, output_shape=deconv_1u_shape, strides=DECONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="deconvolution_1_up")
    concat_1u = tf.concat(values=[deconv_1u, relu_1d], axis=3, name="concat_1_up")
    conv_1u = tf.nn.conv2d(input=concat_1u, filter=weight_conv_1u, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_1_up")
    relu_1u = tf.nn.relu(features=conv_1u + bias_conv_1u, name="relu_1_up")
    if print_shapes:
        print("1 Up: ", relu_1u.shape)

    # Level 0 Up
    weight_conv_0u = tf.Variable(tf.truncated_normal([CONV_FILTER_SIZE, CONV_FILTER_SIZE, CONV_1_SIZE, 1], stddev=DEFAULT_DEV))
    weight_deconv_0u = tf.Variable(tf.truncated_normal([DECONV_FILTER_SIZE, DECONV_FILTER_SIZE, CONV_0_SIZE, CONV_1_SIZE], stddev=DEFAULT_DEV))
    bias_conv_0u = tf.Variable(tf.constant(DEFAULT_DEV, shape=[1]))
    deconv_0u_shape = tf.stack([batch_size, int(relu_1u.shape[1]*2), int(relu_1u.shape[2]*2), CONV_0_SIZE])
    deconv_0u = tf.nn.conv2d_transpose(value=relu_1u, filter=weight_deconv_0u, output_shape=deconv_0u_shape, strides=DECONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="deconvolution_0_up")
    concat_0u = tf.concat(values=[deconv_0u, relu_0d], axis=3, name="concat_0_up")
    conv_0u = tf.nn.conv2d(input=concat_0u, filter=weight_conv_0u, strides=CONV_STRIDES, padding=PADDING_CONFIG, data_format=DATA_FORMAT_CONFIG, name="convolution_0_up")
    result = tf.nn.relu(features=conv_0u + bias_conv_0u, name="relu_0_up")
    transposed_result = tf.transpose(result, [0, 2, 1, 3], name="end_transpose")
    if print_shapes:
        print("Result: ", transposed_result.shape)

    return transposed_result, image_data, ground_truth
