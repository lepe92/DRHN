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

### Input of the script
* N denotes the size of the residual blocks
* H denotes the number of units in the Hash Layer.
Also, you need to indicate in the method load_data the path to the folder containing the dataset.

### Output of the script
The program will generate different files during execution.
* cifar10_deep_residual_hashing_n_k.npz - This file is contain all the weigths of the model, so you can just load the model instead of training again.
* cifar10_n_hashk_codes.pkl - Contain the index generated for the training set.
* cifar10_n_hashk_test_codes.pkl - Contain the index generated for the test set.
* map_n_k.pkl - Contain the MAP value of the evaluation.
* precision_at_k_n_k.pkl - Contain the precision when K elements are returned in a query, you can use this values to generate the plot of Precision against Number of Images Retrieved as is presented in the extended version of the paper.
In all the files after n and k will be the corresponding number of that parameter.

## Publications
The extended abstract version of the paper is available at https://doi.org/10.1007/978-3-319-68612-7.

## Authors
* **Edwin Jimenez-Lepe** `eejimenez@gdl.cinvestav.mx`
* **Andrez Mendez-Vazquez** `amendez@gdl.cinvestav.mx`

## License
This project is licensed under the Apache License, Version 2.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
* Centro de Investigación y de Estudios Avanzados del Instituto Politécnico Nacional [CINVESTAV - Unidad Guadalajara](http://www.gdl.cinvestav.mx/)
* Consejo Nacional de Ciencia y Tecnología [CONACYT](https://www.conacyt.gob.mx/)
* Lin, K., Yang, H. F., Hsiao, J. H., & Chen, C. S. (2015). Deep learning of binary hash codes for fast image retrieval. 
* Theano Development Team
* Lasagne Community for their recipe of Deep Residual Learning.
* European Neural Network Society (ENNS)
