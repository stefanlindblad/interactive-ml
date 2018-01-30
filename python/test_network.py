import tensorflow as tf
import numpy as np
from nnao_network import build_nnao_network
from helper.load_data import read_input_data, read_input_file
from helper.save_data import write_output_file

def test_network(input_file, output_file, checkpoint_path):

    R, G, B, D = read_input_file(input_file)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    output = sess.run([result], feed_dict={image_data: read_input_data(0.1, 1000, R, G, B, D)})
    slim_output = np.squeeze(output)
    write_output_file(output_file, slim_output)

####################################

# Creation of the network and training structures
sess = tf.Session()
result, image_data, _ = build_nnao_network(print_shapes=False)

input_file = "Input_108.exr"
output_file = "Output_108.exr"
checkpoint_path = "checkpoints/train_nnao_29_1_20_34_epoch_12.ckpt"
test_network(input_file, output_file, checkpoint_path)

