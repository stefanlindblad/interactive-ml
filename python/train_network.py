import tensorflow as tf
import numpy as np
from nnao_network import build_nnao_network
from helper.load_data import read_input_data, read_truth_data, read_input_file, read_truth_file
from helper.save_data import freeze_graph, write_output_file, write_output_array
from helper.constants import get_width, get_height, get_channels

import calendar
import time

IMG_WIDTH = get_width()
IMG_HEIGHT = get_height()
IMG_CHANNELS = get_channels()

sess = tf.Session()
result, image_data, ground_truth = build_nnao_network(print_shapes=False)

# Training Stuff
mse = tf.losses.mean_squared_error(labels=ground_truth, predictions=result)
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)
#sess.run(tf.global_variables_initializer())

error, target = 1, 0.0001
epoch, max_epochs = 0, 30000
count = 0

R,G,B,D = read_input_file("Input.exr")
T = read_truth_file("AO.exr")

saver = tf.train.Saver()
saver.restore(sess, "checkpoints/one_img_1516667685.ckpt")

#tf.train.write_graph(sess.graph_def, '.', 'nnao_graph.pbtxt') 
#freeze_graph("nnao_graph.pbtxt", "checkpoints/tmp1515937755.ckpt", "frozen_1515937755.pb")

# Training Loop
while error > target and epoch < max_epochs:
    error, _ = sess.run([mse, train], feed_dict={image_data: read_input_data(0.1, 1000.0, R, G, B, D), ground_truth: read_truth_data(T)})
    epoch += 1
    print("Done with Epoch: [", epoch, "/", max_epochs, "] with an MSE of: ", error)
    
saver = tf.train.Saver()
ts = calendar.timegm(time.gmtime())
path = "checkpoints/one_img_" + str(ts) + ".ckpt"
save_path = saver.save(sess, path)
print("epoch: ", epoch, "mse: ", error, "saved as: ", path)