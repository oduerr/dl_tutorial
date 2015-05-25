# Installation on Top of Data Science Toolbox
The probably easiest way to install a working copy of nolearn lasagne is to use the data science toolbox from [http://datasciencetoolbox.org/](http://datasciencetoolbox.org/) which can be installed as a virtual machine or on Amazon. Please make sure that you also set up ipython notebook support (step 5) in the tutorial.

## Adding support for nolearn
After you installed the data-science toolbox as described above. Log on the box (`vagrant ssh`) and install the following additional packages

* Update ipython to version 3.0 and the missing jsonschema library via
```
vagrant@data-science-toolbox:~$ sudo pip install --upgrade ipython 
vagrant@data-science-toolbox:~$ sudo pip install jsonschema
```

* Install nolearn, lasagne and theano via:
```
vagrant@data-science-toolbox:~$ sudo pip install  -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
```

## Clone the repository
In the standard data-science toolbox configuration the notebooks are stored in `/home/vagrant/notebooks` go into that directory and clone this repository.
```
  vagrant@data-science-toolbox:~$ cd notebooks/
  vagrant@data-science-toolbox:~$ git clone https://github.com/oduerr/dl_tutorial.git
```

## Done
You can now start-up the ipython notebook server with
```
  vagrant@data-science-toolbox:~$ sudo ipython notebook --profile=dst
```
And log in from your local machine using [https://localhost:8888/](https://localhost:8888/)









