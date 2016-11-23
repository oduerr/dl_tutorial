
# coding: utf-8

# ## Debugging with python
# This notebook show how, you can use the mechanism to embed python code for debugging. The idea is taken from https://wookayin.github.io/TensorflowKR-2016-talk-debugging/#53
# 
# We show an example for linear regression


import tensorflow as tf
import numpy as np
N = 30
x_data = (np.linspace(0,10,N)).astype('float32')
y_data = (2.42 * x_data + 0.42 + np.random.normal(0,1,N)).astype('float32')


tf.reset_default_graph()
# Defining the graph (construction phase)
a = tf.Variable(1.0, name = 'a') #Note that 1.0 is needed
b = tf.Variable(0.01, name = 'b')
x = tf.placeholder('float32', [N], name='x_data')
y = tf.placeholder('float32', [N], name='y_data')
loss = tf.reduce_sum(tf.square(a*x + b - y)) #Sum is called reduce_sum 


def _debug_print_func(loss_val, a_val, b_val, x_val, y_val):
    print 'In debug print: Loss {} (a={}, b={} x_val.shape {}'.format(loss_val, a_val, b_val, x_val.shape)
    return False


# We now have to loop the into the graph. We do this by decorating the loss function.
# In[28]:
debug_op = tf.py_func(_debug_print_func, [loss, a, b, x, y], [tf.bool])
with tf.control_dependencies(debug_op): 
    loss = tf.identity(loss, name='out')


train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
epochs = 5
results = np.zeros((epochs, 4))
init_op = tf.initialize_all_variables() #Initialization
with tf.Session() as sess:
    sess.run(init_op) #Running the initialization
    for e in range(epochs): #Fitting the data for some epochs
        res = sess.run([train_op, loss, a, b], feed_dict={x:x_data, y:y_data})  
        results[e] = res
    res = sess.run([loss, a, b], feed_dict={x:x_data, y:y_data})



