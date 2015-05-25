## Tutorial for lasagne
The following notebooks are tutorials, demonstrating how easy it is to do deep learning with nolearn/lasagne library. Some basic knowledge of deep learning is assumed. You should know multilayer perceptions and convolutional neural networks. See [http://deeplearning.net/tutorial/](http://deeplearning.net/tutorial/) for an introduction. 

* [MinimalLasagneMLP.ipynb](MinimalLasagneMLP.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/MinimalLasagneMLP.ipynb) shows how to create a multilayer perceptron.
* [MinimalLasagneCNN.ipynb](MinimalLasagneCNN.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/MinimalLasagneCNN.ipynb) shows how to create a (simple) convolution neural network
* [DataAugmentation.ipynb](DataAugmentation.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/DataAugmentation.ipynb) shows how to do training data augmentation (in principle)
* [DataAugmentationII.ipynb](DataAugmentationII.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/DataAugmentationII.ipynb) shows how to use training data augmentation to ease overfitting and do a better prediction.

## Installation
To install the required python packages such as nolearn,lasagne and theano do
```
pip install -r https://raw.githubusercontent.com/oduerr/dl_tutorial/master/lasagne/requirements.txt
```
### Installation on VM or Amazon
Alternatively you can use a VPN. For example [http://datasciencetoolbox.org/](http://datasciencetoolbox.org/) provides a VM and Amazon AMIs to on top of the data science toolbox see [README_DataScience_ToolBox.md](README_DataScience_ToolBox.md) for a step-by-step instruction.


#### Other tutorials. 
Creating this tutorial I was very much inspired by the following great tutorials:

* [danielnouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) describing the use of Lasagne in facial keypoint detection
* [msegala Code Plankton](https://github.com/msegala/Kaggle-National_Data_Science_Bowl) quite simple code, which also a contains a `Batchiterator` 

#### Other inks
* [Lasagne Github Page](https://github.com/Lasagne/Lasagne) good starting point, see also the [mailing list](https://groups.google.com/forum/#!forum/lasagne-users)
* [Winning Solution for the plankton challenge](http://benanne.github.io/2015/03/17/plankton.html) from the creator of lasagne
