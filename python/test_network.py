import tensorflow as tf
import numpy as np
from nnao_network import build_nnao_network
from helper.load_data import read_input_data, read_truth_data, read_input_file, read_truth_file
from helper.save_data import write_output_file

def test_network(input_file, truth_file, output_file, checkpoint_path):

    R, G, B, D = read_input_file(input_file)
    T = read_truth_file(truth_file)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    output, error = sess.run([result, mse], feed_dict={image_data: read_input_data(0.1, 1000, R, G, B, D), ground_truth: read_truth_data(T)})
    slim_output = np.squeeze(output)
    write_output_file(output_file, slim_output)
    return error

####################################

# Creation of the network and training structures
sess = tf.Session()
result, image_data, ground_truth = build_nnao_network(print_shapes=False)
mse = tf.losses.mean_squared_error(labels=ground_truth, predictions=result)

input_file = "Input_LA.exr"
truth_file = "AO_LA.exr"
output_file = "Output_LA.exr"
checkpoint_path = "checkpoints/train_nnao_30_1_13_30_epoch_20.ckpt"
print(test_network(input_file, truth_file, output_file, checkpoint_path))

