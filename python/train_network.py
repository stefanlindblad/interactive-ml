import tensorflow as tf
from tensorflow.python.layers import utils
from tensorflow.python.tools import freeze_graph
from nnao_network import build_nnao_network
from helper.load_data import read_input_data, read_truth_data, read_input_file, read_truth_file
from helper.constants import get_width, get_height, get_channels

import calendar
import time

IMG_WIDTH = get_width()
IMG_HEIGHT = get_height()
IMG_CHANNELS = get_channels()


result, image_data, ground_truth = build_nnao_network(print_shapes=False)
mse = tf.losses.mean_squared_error(labels=ground_truth, predictions=result)
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

error, target = 1, 0.015
epoch, max_epochs = 0, 5000

R,G,B,D = read_input_file("Input.exr")
T = read_truth_file("AO.exr")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

while error > target and epoch < max_epochs:
    #saver.restore(sess, "checkpoints/tmp263.ckpt")
    epoch += 1
    error, _ = sess.run([mse, train], feed_dict={image_data: read_input_data(0.1, 1000.0, R, G, B, D), ground_truth: read_truth_data(T)})
    print("Done with running Epoch: [", epoch, "/", max_epochs, "] with an MSE of: ", error)

saver = tf.train.Saver()
ts = calendar.timegm(time.gmtime())
path = "checkpoints/tmp" + str(ts) + ".ckpt"
save_path = saver.save(sess, path)
print("epoch: ", epoch, "mse: ", error, "saved as: ", path)