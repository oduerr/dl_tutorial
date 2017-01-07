import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def beates_dropout(input_vals, keep_prob = 0.5):
    # Dropout
    keep_prob = ops.convert_to_tensor(keep_prob, dtype=input_vals.dtype, name="keep_prob")
    noise_shape = array_ops.shape(input_vals)
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(noise_shape, dtype=input_vals.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    ret = input_vals * binary_tensor
    ret.set_shape(input_vals.get_shape())
    return ret


if __name__ == '__main__':
    LOG_DIR = '/tmp/dumm'
    N =  10 #Samples
    p =   5 #Features
    X = np.asarray(np.reshape(np.random.normal(0,1, N*p), (N,p)), dtype='float32')

    print(tf.__version__)
    input_vals = tf.placeholder('float32', (None, p))
    tensor_of_interest = beates_dropout(input_vals, 0.5) #For example a hidden layer such as FC1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor_of_interest, feed_dict={input_vals : X})



    ## Writing out the graph
    graph = tf.get_default_graph();
    output_graph_def = graph.as_graph_def()
    with tf.gfile.GFile('/tmp/test.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    ## reading the graph
    tf.reset_default_graph()
    with tf.gfile.GFile('/tmp/test.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,  name='')

    input_vals = tf.placeholder('float32', (None, p))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(tensor_of_interest, feed_dict={input_vals : X})
