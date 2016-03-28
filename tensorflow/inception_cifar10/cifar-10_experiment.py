import os
import time
from classify_image import *
import cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def extract_features(file_name):
    d = unpickle(os.path.join(os.path.expanduser('~/Dropbox/data/CIFAR-10/cifar-10-batches-py/'), file_name))
    data = d['data']
    imgs = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
    y = np.asarray(d['labels'], dtype='uint8')

    FLAGS.model_dir = 'model/'
    maybe_download_and_extract()
    create_graph()
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        predictions = np.zeros((len(y), 1008), dtype='float32')
        representations = np.zeros((len(y), 2048), dtype='float32')
        for i in range(len(y)):
            start = time.time()
            [reps, preds] = sess.run([representation_tensor, softmax_tensor], {'DecodeJpeg:0': imgs[i]})
            if (i % 10 == 0):
                print("{}/{} Time for batch {} ".format(i, len(y), time.time() - start))
            predictions[i] = np.squeeze(preds)
            representations[i] = np.squeeze(reps)
        np.savez_compressed(file_name + ".npz", predictions=predictions, representations=representations, y=y)

if __name__ == '__main__':
    extract_features('test_batch')
    extract_features('data_batch_1')
    extract_features('data_batch_2')
    extract_features('data_batch_3')
    extract_features('data_batch_4')
    extract_features('data_batch_5')




