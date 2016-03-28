## Tutorial for nolearn.lasagne
The following notebooks are tutorials, demonstrating how easy it is to do deep learning with nolearn/lasagne library. Some basic knowledge of deep learning is assumed. You should know multilayer perceptions and convolutional neural networks. See [http://deeplearning.net/tutorial/](http://deeplearning.net/tutorial/) or [cs231n](http://cs231n.github.io/convolutional-networks/) for an introduction. 

* [MinimalLasagneMLP.ipynb](MinimalLasagneMLP.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/MinimalLasagneMLP.ipynb) shows how to create a multilayer perceptron.
* [MinimalLasagneCNN.ipynb](MinimalLasagneCNN.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/MinimalLasagneCNN.ipynb) shows how to create a (simple) convolution neural network
* [DataAugmentation.ipynb](DataAugmentation.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/DataAugmentation.ipynb) shows how to do training data augmentation (in principle)
* [DataAugmentationII.ipynb](DataAugmentationII.ipynb) [nbviewer](http://nbviewer.ipython.org/github/oduerr/dl_tutorial/blob/master/lasagne/DataAugmentationII.ipynb) shows how to use training data augmentation to ease overfitting and do a better prediction.

Slides of a tutorial on these notebooks given at the Zurich Machine Learning Meetup can be found 
[here](https://dl.dropboxusercontent.com/u/9154523/talks/ConvNets_ZH_ML.pptx.pdf)

## Installation
To install the required python packages follow the installation procedure descriped in https://github.com/dnouri/nolearn

To clone the demo itself
```
git clone https://github.com/oduerr/dl_tutorial.git
```
To start the ipython notebook server
```
ipython notebook
```

### Installation on VM or Amazon
Alternatively you can use a VM. For example [http://datasciencetoolbox.org/](http://datasciencetoolbox.org/) provides a VM and Amazon AMIs. To install the data science toolbox and lasagne on top of it see [README_DataScience_ToolBox.md](README_DataScience_ToolBox.md) for a step-by-step instruction.


#### Other tutorials. 
Creating this tutorial I was very much inspired by the following great tutorials:

* [danielnouri](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) describing the use of Lasagne in facial keypoint detection. A great tutorial which inspired this tutorial.

* [msegala Code Plankton](https://github.com/msegala/Kaggle-National_Data_Science_Bowl) quite simple code, which also a contains a `Batchiterator` 

#### Other inks
* [Lasagne Github Page](https://github.com/Lasagne/Lasagne) good starting point, see also the [mailing list](https://groups.google.com/forum/#!forum/lasagne-users)
* [Winning Solution for the plankton challenge](http://benanne.github.io/2015/03/17/plankton.html) from the creator of lasagne
