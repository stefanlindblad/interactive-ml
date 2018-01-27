import tensorflow as tf
from nnao_network import build_nnao_network
from helper.save_data import freeze_graph

def export_frozen_graph(timecode, session):
    interactive_module = tf.load_op_library("C:/code/tensorflow/tensorflow/contrib/cmake/build/Release/interactive_ops.dll")

    graph_path = "nnao_graph.pbtxt"
    checkpoint_path = "checkpoints/one_img_" + str(timecode) + ".ckpt"
    frozen_path = "frozen_" + str(timecode) + ".pb"

    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)
    freeze_graph(graph_path, checkpoint_path, frozen_path)