
# coding: utf-8

# # Tricks of the trade TF/Keras [small dataset]
# 
# In this script we build a small multilayer perceptron with two hidden layers having 500 and 50 neurons each for classifying the MNIST database of handwritten digits using Keras. It uses the full data set better to run on a GPU.
# 
# Below are several experiments.

# In[1]:

import numpy as np

import time
import tensorflow as tf
tf.set_random_seed(1)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
import sys
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def get_data():
    mnist = read_data_sets("../data/", one_hot=True, reshape=True, validation_size=2000)
    X_train = mnist.train.images
    X_val = mnist.validation.images
    Y_train = mnist.train.labels
    Y_val = mnist.validation.labels
    print(X_train.shape, X_train.shape, Y_val.shape, Y_train.shape)
    return X_train, X_val, Y_val, Y_train


# ### Suggestions for the experiment
# 
# Let the experiments run for 100 epochs. You might need to restart the kernel so that namings of the layers are the same
# 
# * with init zero 
# * with sigmoid activation 
# * with ReLU activation
# * with dropout (p=0.3)
# * with batch-normalization and dropout

# In[ ]:

### First model with all zeros
def model_1():
    name = 'sigmoid_init0'
    model = Sequential()
    model.add(Dense(500, batch_input_shape=(None, 784), init='zero'))
    model.add(Activation('sigmoid'))

    model.add(Dense(50,init='zero'))
    model.add(Activation('sigmoid'))

    model.add(Dense(10, activation='softmax',init='zero'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model, name

def model_2():
    name = 'sigmoid'
    model = Sequential()
    model.add(Dense(500, batch_input_shape=(None, 784)))
    #model.add(Dropout(0.3))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('sigmoid'))

    model.add(Dense(50))
    #model.add(Dropout(0.3))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('sigmoid'))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model, name


### Model with default initialization
def model_3():
    name = 'relu'
    model = Sequential()
    model.add(Dense(500, batch_input_shape=(None, 784)))
    #model.add(Dropout(0.3))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(50))
    #model.add(Dropout(0.3))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model, name


# In[ ]:

### Model with default initialization
def model_4():
    name = 'dropout'
    model = Sequential()
    model.add(Dense(500, batch_input_shape=(None, 784)))
    model.add(Dropout(0.3))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Dropout(0.3))
    #model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model, name


# In[ ]:

### Model with default initialization 
def model_5():
    name = 'dropout_batch'
    model = Sequential()
    model.add(Dense(500, batch_input_shape=(None, 784)))
    model.add(Dropout(0.3))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Dropout(0.3))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model, name


# In[ ]:

if __name__ == '__main__':
    print(keras.__version__, tf.__version__, sys.version_info)
    from tensorflow.python.platform import flags
    FLAGS = flags.FLAGS
    X_train, X_val, Y_val, Y_train = get_data()
    model, name = model_2()
    print(model.summary())

    log_dir='/notebooks/tensorflow/path_to_fc_nets/tb/' + name


    tensorboard = keras.callbacks.TensorBoard(
        log_dir='/notebooks/tensorflow/path_to_fc_nets/tb_full_mnist/' + name + '/',
        write_graph=True,
        histogram_freq=5
    )
    history = model.fit(X_train,Y_train,
              nb_epoch=1,
              batch_size=128,
              callbacks=[tensorboard],
              validation_data=[X_val, Y_val], verbose=2)

    model.save(name + ".keras")





