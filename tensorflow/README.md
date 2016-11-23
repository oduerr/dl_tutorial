# Collection of examples for learning TensorFlow

## Introduction to TensorFlow 
The following notebooks are also covered in the [introductory presentation](https://home.zhaw.ch/~dueo/bbs/files/TF_Introduction.pdf)
* [linear_regression/01_LineFit.ipynb](linear_regression/01_LineFit.ipynb) A simple line fit 
* [simple_ops/Mandelbrot.ipynb](simple_ops/Mandelbrot.ipynb) Drawing the Madelbrot set using TensorFlow
* [Control_Flow/Mandelbrot.ipynb](Control_Flow/Mandelbrot.ipynb) Explains tensorflow loops and applies them to draw the Madelbrot set

## Advanced Topics
The following notebooks deal with more deep learning relate aspects of TF and are partly convered in this [presentation](https://home.zhaw.ch/~dueo/bbs/files/TF_DeepLearning.pptx.pdf)

### Building networks, accessing them
* [Building_Nice_Networks/Scoping.ipynb](Building_Nice_Networks/Scoping.ipynb) 
Use variable scopes to build networks from building blocks.

* [linear_regression/03_checkpointing.ipynb](linear_regression/03_checkpointing.ipynb) demonstrates how to checkpoint and reload a model

* [stored_models/Using_Trained_Nets.ipynb](stored_models/Using_Trained_Nets.ipynb) Load a trained network, invastigate the network to find relevant entry / exit points. Use these to feed data through the network (for classification of novel images). The notebook further shows how to get variables like the kernels from a CNN from the network.

* [stored_models/Finetuning.ipynb](stored_models/Finetuning.ipynb) Setting up a network with TF-Slim and do transfer learning 

### Debugging networks
* [linear_regression/02_Inspecting_the_graph.ipynb](linear_regression/02_Inspecting_the_graph.ipynb) How to use summaries and visualize them in tensorboard

* [debugging/print](debugging/print.ipynb) Explanation of the `tf.Print()` function

* [debugging/debug_with_python.ipynb](debugging/debug_with_python.ipynb) Debugging by embedding python code


# Miscelaneous things with TensorFlow
* [vae](vae/) contains examples of Variational Autoencoder VAE

* [faces](faces/) shows how to transfer a network for face recognition from caffee to tensorflow

* [inception_cifar10](inception_cifar10/) Shows how to extract features from a network trained on ImageNet and applies them on CIFAR_10.



