# DRHN
Implementation of Deep Residual Hashing Network for Image Retrieval.

## Getting Started
This code is ready to run, is a python script that will load the cifar-10 dataset (please look at the method load_data) and train for classification task using an initial learning rate of 0.1 for 50 epochs. After the training step the program will generate the binary codes for the images and perform the evaluation for Image Retrieval.

### Prerequisites
To use the program you need [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html), [Theano](http://deeplearning.net/software/theano/) and [Lasagne](https://lasagne.readthedocs.io/en/latest/). We highly recommend you to use a  [Docker](https://www.docker.com/) image that already have the configuration for that tools.

## Running the file
Once that you have the correct configuration of Theano to use your graphic card, you can run the program with the next command.
```
$THEANO_FLAGS=floatX=float32,device=gpu0 python DRHN.py N H
```
Where N denotes the size of the residual blocks and H denotes the number of units in the Hash Layer.

### Input of the script
### Output of the script

## Authors
