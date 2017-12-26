import tensorflow as tf

interactive_module = tf.load_op_library("C:/code/tensorflow/tensorflow/contrib/cmake/build/Release/interactive_ops.dll")

with tf.Session() as sess:
    with tf.device("/device:GPU:0"):
        reader = tf.placeholder(tf.float32, shape=(1, None, None, 4), name="input")
        interactivated = interactive_module.interactive_input(reader)
        contrast = tf.image.adjust_contrast(interactivated, 1.01)
        outervated = interactive_module.interactive_output(contrast)
        tf.train.write_graph(sess.graph_def, "C:/code/interactive-ml/python/", "editing.pb", True)