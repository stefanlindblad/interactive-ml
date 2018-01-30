import tensorflow as tf
import numpy as np
from nnao_network import build_nnao_network
from export_network import export_frozen_graph
from helper.load_data import read_input_data, read_truth_data, read_input_file, read_truth_file, read_input_set, read_truth_set
from helper.save_data import write_output_file, write_output_array, write_output_array2
import datetime

def train_network(from_beginning, min_error, data_set_size, epochs, models, train_path, checkpoint_path = ""):
    if from_beginning:
        sess.run(tf.global_variables_initializer())
    else:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

    error = 1
    epoch = 12
    count = 0

    # Training Loop
    while error > min_error and epoch < epochs:
        while count < data_set_size:
            error, _ = sess.run([mse, train], feed_dict={image_data: read_input_set(models, count, train_path), ground_truth: read_truth_set(models, count, train_path)})
            count += 1
            if count % 20 == 0:
                saver = tf.train.Saver()
                now = datetime.datetime.now()
                path = "checkpoints/train_nnao_" + str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "_" + str(now.minute) + "_count_" + str(count) + ".ckpt"
                saver.save(sess, path)
                status_string = "Count: " + str(count) + " with MSE of: " + str(error) + " saved as: " + str(path)
                print(status_string.encode("utf-8").decode("ascii"))
            count_string = "Done with Count: [" + str(count) + "/" + str(data_set_size) + "] with an MSE of: " + str(error)
            print(count_string.encode("utf-8").decode("ascii"))
        count = 0
        epoch += 1
        saver = tf.train.Saver()
        now = datetime.datetime.now()
        path = "checkpoints/train_nnao_" + str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "_" + str(now.minute) + "_epoch_" + str(epoch) + ".ckpt"
        saver.save(sess, path.encode("utf-8").decode("ascii"))
        epoch_string = "Epoch: " + str(epoch) + " with MSE of: " + str(error) + " saved as: " + path
        print(epoch_string.encode("utf-8").decode("ascii"))

####################################

# Creation of the network and training structures
sess = tf.Session()
result, image_data, ground_truth = build_nnao_network(print_shapes=False)
mse = tf.losses.mean_squared_error(labels=ground_truth, predictions=result)
train = tf.train.GradientDescentOptimizer(0.005).minimize(mse)

learn_models = "learn_models.txt"
train_data = "D:/train_data/"
checkpoint_path = "checkpoints/train_nnao_29_1_20_34_epoch_12.ckpt"
train_network(False, 0.0001, 600, 20, learn_models.encode("utf-8").decode("ascii"), train_data.encode("utf-8").decode("ascii"), checkpoint_path.encode("utf-8").decode("ascii"))

#tf.train.write_graph(sess.graph_def, '.', 'nnao_graph.pbtxt')
#export_frozen_graph(timecode, sess)
