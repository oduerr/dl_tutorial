import tensorflow as tf
import numpy as np

# 
if __name__ == '__main__':
    N = 30
    x_data = (np.linspace(0,10,N)).astype('float32')
    y_data = (2.42 * x_data + 0.42 + np.random.normal(0,1,N)).astype('float32')
    a = tf.Variable(1.0, name = 'a') #Note that 1.0 is needed
    b = tf.Variable(0.01, name = 'b')
    x = tf.placeholder('float32', [N], name='x_data')
    y = tf.placeholder('float32', [N], name='y_data')


    resi = a*x + b - y
    loss = tf.reduce_sum(tf.square(resi))
    init_op = tf.initialize_all_variables() #Initialization

    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    epochs = 100

    tf.name_scope("fitting")
    loss_summary = tf.scalar_summary("loss_summary", loss) #<---
    resi_summart = tf.histogram_summary("residual_summart", resi)
    merged_summary_op = tf.merge_all_summaries()   #<-----  before the session
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.train.SummaryWriter("/tmp/dumm", sess.graph_def, 'graph.pbtxt')
        for e in range(epochs): #Fitting the data for 10 epochs
            sess.run(train_op, feed_dict={x:x_data, y:y_data})
            #print(sess.run(loss, feed_dict={x:x_data, y:y_data}))
            sum_str = sess.run(merged_summary_op, feed_dict={x:x_data, y:y_data})
            writer.add_summary(sum_str, e)
        res = sess.run([loss, a, b], feed_dict={x:x_data, y:y_data})
        print(res)
    print('Finished all')