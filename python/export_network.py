import tensorflow as tf
from nnao_network import build_nnao_network
from helper.save_data import freeze_graph
from helper.constants import get_width, get_height

IMG_WIDTH = get_width()
IMG_HEIGHT = get_height()

def export_frozen_graph(identifier, session):
    interactive_module = tf.load_op_library("C:/code/tensorflow/tensorflow/contrib/cmake/build/Release/interactive_ops.dll")
    print(identifier)
    graph_path = "nnao_graph_" + identifier + ".pbtxt"
    checkpoint_path = "checkpoints/train_nnao_30_1_13_30_epoch_20.ckpt"
    frozen_path = "frozen_" + identifier + ".pb"

    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)
    freeze_graph(graph_path, checkpoint_path, frozen_path)


#======================

sess = tf.Session()
result, _, _ = build_nnao_network(print_shapes=True)
export_frozen_graph(str(IMG_WIDTH) + "x" + str(IMG_HEIGHT), sess)
#tf.train.write_graph(sess.graph_def, '.', 'nnao_graph_' + str(IMG_WIDTH) + "x" + str(IMG_HEIGHT) + '.pbtxt')
