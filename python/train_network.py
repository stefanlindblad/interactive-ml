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
    epoch = 0
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
                print("Count: ", Count, "with MSE of: ", error, "saved as: ", path)
            print("Done with Count: [", count, "/", data_set_size, "] with an MSE of: ", error)
        count = 0
        epoch += 1
        saver = tf.train.Saver()
        now = datetime.datetime.now()
        path = "checkpoints/train_nnao_" + str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "_" + str(now.minute) + "_epoch_" + str(epoch) + ".ckpt"
        saver.save(sess, path)
        print("Epoch: ", epoch, "with MSE of: ", error, "saved as: ", path)


####################################

# Creation of the network and training structures
sess = tf.Session()
result, image_data, ground_truth = build_nnao_network(print_shapes=False)
mse = tf.losses.mean_squared_error(labels=ground_truth, predictions=result)
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

train_graph(True, 0.001, 600, 10, "learn_models.txt", "D:/train_data/")

#tf.train.write_graph(sess.graph_def, '.', 'nnao_graph.pbtxt')
#export_frozen_graph(timecode, sess)
