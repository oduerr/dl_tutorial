import tensorflow as tf
import numpy as np

tf.reset_default_graph()
x = tf.placeholder(tf.float32, name='X_')
y = tf.placeholder(tf.float32, name='Y_')
b = tf.Variable(1.0)
y_pred = 2 * x + b     # x -> 2*x + b
loss = (y - y_pred)**2

loss = tf.Print(loss, [y, y_pred, loss],
               message='Debug y, y_pred, loss ', name='Debug_Print', first_n=5)
# Not working is
# tf.Print(loss, [y, y_pred, (y-y_pred)**2], name='Name', first_n=5)
# loss = tf.Print(loss, y, name='Name', first_n=5)

tf.train.SummaryWriter('/tmp/dumm/debug_print', tf.get_default_graph()).close()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print('Loss : {}'.format(sess.run(loss, feed_dict={x:10, y:10})))

