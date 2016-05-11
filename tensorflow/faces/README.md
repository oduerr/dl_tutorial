The notebook (vgg_faces.ipynb) shows how to load the vgg_faces network (http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) into tensorflow and predicing the identity of a given celibrity.

### Creating the TensorFlow network
The network has been tranfered from caffe using the following steps:

* Installing the latest caffe version
The problem is that the transfer tool (see below) needs a new version of the cafe prototypes. Instead of installing caffe I did install a caffe-docker container 'kaixhin/caffe' (on a Ubuntu 14.04 box).

* Downloading a docker container with caffe preinstalled
```
> sudo docker pull kaixhin/caffe 
```

* Starting a caffe docker container
The code below maps the directory containing the old `/home/dueo/Dropbox/Server/` caffe model to '/opt/data1'
```
> sudo docker run --privileged=true -v /home/dueo/Dropbox/Server/:/opt/data1 -it kaixhin/caffe bash
```

* Converting the prototype (in the docker container)
```
root@5d4bb5d1f0cf:~/caffe# ./build/tools/upgrade_net_proto_text /opt/data1/vgg_face_caffe/VGG_FACE_deploy.prototxt /opt/data1/vgg_face_caffe/VGG_FACE_deploy_new.prototxt 
```

* Cloning the conversion tool (in my Ubuntubox)
```
> git clone https://github.com/ethereon/caffe-tensorflow.git
```

* Transfering the caffe model
```
> ~/workspace_other/caffe-tensorflow$ ./convert.py /home/dueo/Dropbox/Server/vgg_face_caffe/VGG_FACE_deploy_new.prototxt --code-output-path=/home/dueo/Dropbox/Server/vgg_face_tf/vgg_face.py

> ./convert.py /home/dueo/Dropbox/Server/vgg_face_caffe/VGG_FACE_deploy_new.prototxt --caffemodel /home/dueo/Dropbox/Server/vgg_face_caffe/VGG_FACE_new.caffemodel --data-output-path=/home/dueo/Dropbox/Server/vgg_face_tf/vgg_face.npy
```
