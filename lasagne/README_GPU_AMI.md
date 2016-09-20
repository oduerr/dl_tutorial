# Installation on Top of public AMI

If you want full GPU support. You can use a AMI see provided by Markus Beissinger. This public AMI can be found at: ami-b141a2f5 (Theano - CUDA 7) in the N. California region. See [http://markus.com/install-theano-on-aws/](http://markus.com/install-theano-on-aws/) for details. You need a g2.2xlarge instance (cost approx $0.702 / h) or use a spot-instance.

```
sudo apt-get install libfreetype6-dev libxft-dev 
sudo pip install -r https://raw.githubusercontent.com/oduerr/dl_tutorial/master/lasagne/requirements.txt
```

## To set up a nbserver 
The AMI needs a notebook server see e.g. 
http://badhessian.org/2013/11/cluster-computing-for-027hr-using-amazon-ec2-and-ipython-notebook/





