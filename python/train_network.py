import tensorflow as tf
import numpy as np
from nnao_network import build_nnao_network
from export_network import export_frozen_graph
from helper.load_data import read_input_data, read_truth_data, read_input_file, read_truth_file
from helper.save_data import write_output_file, write_output_array, write_output_array2
import calendar
import time

def train_graph(new_epoch, checkpoint_path, min_error, max_epochs):
    if new_epoch:
        sess.run(tf.global_variables_initializer())
    else:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

    error, target = 1, min_error
    epoch = 0
    count = 0

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


sess = tf.Session()
result, image_data, ground_truth = build_nnao_network(print_shapes=False)
#mse = tf.losses.mean_squared_error(labels=ground_truth, predictions=result)
#train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

timecode = 1516849221

R,G,B,D = read_input_file("Input_pot_512.exr")
T = read_truth_file("AO_pot_512.exr")

checkpoint_path = "checkpoints/one_img_" + str(timecode) + ".ckpt"

#tf.train.write_graph(sess.graph_def, '.', 'nnao_graph.pbtxt')

export_frozen_graph(timecode, sess)

#train_graph(True, checkpoint_path, 0.05, 1000)


#saver = tf.train.Saver()
#saver.restore(sess, checkpoint_path)
#data = sess.run([result], feed_dict={image_data: read_input_data(0.1, 1000.0, R, G, B, D)})
#write_output_file("test.exr", np.squeeze(data))
