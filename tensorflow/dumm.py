import numpy as np
import tensorflow as tf

tf.nn.dynamic_rnn()

def gen_data(size=1000000):
    Xs = np.array(np.random.choice(3, size=(size,))) #Random Weather
    Y = []
    ice = 2 #Our stock of icecream at start
    for t,x in enumerate(Xs):
        if (t - 2) >= 0 and Xs[t - 2] == 1 and ice < 2: #claudy, and not full
            ice += 1
        if x == 0: # It is sunny we therefore sell ice, if we have
            if ice > 0:
                ice -= 1
        if ice < 0:
            ice = 0
        if ice > 0: #We are not out of stock
            Y.append(1)
        else:
            Y.append(0)
        print ice
    return Xs, np.array(Y)

if __name__ == '__main__':
    gen_data(50)