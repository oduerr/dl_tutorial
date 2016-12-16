import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

LOG_DIR = '/tmp/dumm'
N = 10000 #Samples
p =   10 #Features
X = np.asarray(np.reshape(np.random.normal(0,1, N*p), (N,p)), dtype='float32')
print(X.shape)
print(X.min())
metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
metadata_file.write('Name\tClass\n')
for ll in range(N):
    y = 1
    if (X[ll, 1] > 0):
        y = 0
    metadata_file.write('%06d\t%d\n' % (ll, y))
metadata_file.close()

print(tf.__version__) #Need > 0.12

# Network definition
# A supper simple network
input_vals = tf.placeholder('float32', (None, p))
tensor_of_interest = tf.identity(input_vals) #For example a hidden layer such as FC1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train the network and ...
    _ = sess.run(tensor_of_interest, feed_dict={input_vals : X})
    # much is going on here

    # Now we are ready to write down the embedding...
    EMB = np.zeros((N, p), dtype='float32')
    for i in range(N): #Of course you could do mini-batches
        EMB[i] = sess.run(tensor_of_interest, feed_dict={input_vals: X[i:i+1,:]})
    # The embedding variable, which needs to be stored
    # Note this must Variable not a Tensor
    embedding_var = tf.Variable(EMB,  name='Variable_of_interest')
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(LOG_DIR, 'model2.ckpt'), 1)
