## Collection of examples to learning TensorFlow

## Using TensorFlow for computation
The following notebooks are also convered in the [presentation](https://home.zhaw.ch/~dueo/bbs/files/TF_Introduction.pdf)
* [linear_regression/01_LineFit.ipynb](linear_regression/01_LineFit.ipynb) A simple line fit
* [simple_ops/Mandelbrot.ipynb](simple_ops/Mandelbrot.ipynb) Drawing the Madelbrot set using TensorFlow
* [Control_Flow/Mandelbrot.ipynb](Control_Flow/Mandelbrot.ipynb) Explains tensorflow loops and applies them to draw the madelbrot set

## Using TensorFlow for Deep Learning
### Building networks, accessing them
* [Building_Nice_Networks/Scoping.ipynb](Building_Nice_Networks/Scoping.ipynb) 
Use variable scopes to build networks from building blocks.

* [stored_models/Using_Trained_Nets.ipynb](stored_models/Using_Trained_Nets.ipynb) Load a trained network, invastigate the network to find relevant entry / exit points. Use these to feed data throug the network (e.g. for example make classification on images). It further show how to get variables from the network.

* [stored_models/Finetuning.ipynb](stored_models/Finetuning.ipynb) How to do transfer learning with TensorFlow

### Debugging networks
* [linear_regression/02_Inspecting_the_graph.ipynb](linear_regression/02_Inspecting_the_graph.ipynb) How to use summaries and visualize them in tensorboard
* [debugging/print](debugging/print.ipynb) Explanation of the `tf.Print()` function



