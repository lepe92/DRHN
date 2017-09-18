#!/usr/bin/env python

"""
Based on the Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)
https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
"""
from __future__ import print_function

import matplotlib.pyplot as plt
from itertools import product
from lasagne.layers import get_output


import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in current working directory

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

# ##################### Build the neural network model #######################

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify, sigmoid
from lasagne.layers import batch_norm

def build_cnn(input_var=None, n=5, k=12):
    
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, flip_filters=False,num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))
        
        stack_2 = batch_norm(ConvLayer(stack_1, flip_filters=False,num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu')))
        ##voy aqui, de la linea 132
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l,flip_filters=False, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)#substitute with x

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in,flip_filters=False, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))
    
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):#5
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)#1
    for _ in range(1,n): #4
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    
    ##middle block with no increase_dim
    # 3.1 stack of residual blocks
    for _ in range(n):#5
        l = residual_block(l)

	##middle block with no increase_dim
    # 3.2 stack of residual blocks
    for _ in range(n):#5
        l = residual_block(l)
    
    # 4th stack of residual blocks, output is 128 x 4 x 4
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    # average pooling
    l = GlobalPoolLayer(l,name='globalLayer')
    
    ######insert another layer with k-units where k is the number of bits in the hash
    hash_layer = DenseLayer(
            l, num_units=k,
            W=lasagne.init.HeNormal(),
            nonlinearity=sigmoid,name='hash_layer')
    ############

    # fully connected layer
    network = DenseLayer(
            hash_layer, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper : 
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]

# ############################## Main program ################################
#82 previous
def main(n=5, k=12, num_epochs=50, model=None):
    # Check if cifar data exists
    print ("n= ",n, " k= ",k)

    if not os.path.exists("./cifar-10-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = load_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var, n, k)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    
    if model is None:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = loss + l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(network, trainable=True)
        lr = 0.1
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=sh_lr, momentum=0.9)
        
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    ###function for prediction
    predict_class=theano.function(inputs=[input_var],outputs=test_prediction)

    ####function to generate the related hash-code for the images
    layers = lasagne.layers.get_all_layers(network)
    for l in layers:
        if l.name == 'hash_layer':
            hash_layer_out=l

    prediction_hash=T.switch(T.le(get_output(hash_layer_out, input_var), 0.5), 0., 1.)
    predict_hash = theano.function([input_var], prediction_hash)

    #####################1. TRAINING OR LOADING THE MODEL ########################
    if model is None:
        validation_loss=10
        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # shuffle training data
            train_indices = np.arange(100000)
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices,:,:,:]
            Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, Y_train, 128, shuffle=True, augment=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            if val_err / val_batches < validation_loss:
                validation_loss=val_err / val_batches
                np.savez('cifar10_deep_residual_hashing_n'+str(n)+'_k'+str(k)+'.npz', *lasagne.layers.get_all_param_values(network))
                print ("guardando modelo...")
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch+1) == 41 or (epoch+1) == 61:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        #np.savez('cifar10_deep_residual_model.npz', *lasagne.layers.get_all_param_values(network))
    else:
        # load network weights from model file
        print ('Loading MODEL pre-trained')
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
        validation_of_the_model()

    # Calculate validation error of model:
    def validation_of_the_model():
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    #####################2. GENERATION OF THE CODES ##########################
    def save_obj(obj, name ):
        with open( name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name ):
        with open( name + '.pkl', 'rb') as f:
            return pickle.load(f)

    hashes_training=[]
    index=0

    #generating codes for training images
    for batch in iterate_minibatches(X_train, Y_train, 500, shuffle=False):
        inputs, targets = batch
        pred = predict_hash(inputs).astype(int)

        for element in pred:
            hashes_training.append((index,element,Y_train[index]))
            index+=1
    save_obj(hashes_training,'cifar10_n'+str(n)+'_hash'+str(k)+'k_codes')

    hashes_testing=[]
    index=0
    #generating codes for testing images
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        pred = predict_hash(inputs).astype(int)

        for element in pred:
            hashes_testing.append((index,element,Y_test[index]))
            index+=1
    save_obj(hashes_testing,'cifar10_n'+str(n)+'_hash'+str(k)+'k_test_codes')

    ########################3. MAP evaluation #############################
    #for all elements in test_examples (should be a list of len=10000)
    #MAP evaluation as shown in https://github.com/kevinlin311tw/caffe-cvprw15
    k_=1000
    NS=np.arange(1,k_+1)
    sum_tp = np.zeros(len(NS))
    QueryTimes=10000
    AP=np.zeros(QueryTimes)

    index_of_query=0
    for image_test in hashes_testing:
        for index in range(len(hashes_training)):
            hashes_training[index]=(hashes_training[index][0],hashes_training[index][1],hashes_training[index][2],np.count_nonzero(image_test[1]!=hashes_training[index][1]))#hamming2(image_test[1],hashes_training[index][1]))

        from operator import itemgetter
        hashes_training.sort(key=itemgetter(3))
        
        #comenzamos
        buffer_yes = np.zeros(k_)
        total_relevant = 0
        for i in range(k_):
            #si la etiqueta es igual, sumar
            if hashes_training[i][2]==image_test[2]:
                buffer_yes[i]=1
                total_relevant+=1
        #print (total_relevant)
        P = np.divide(np.cumsum(buffer_yes),NS,dtype=float)
        if np.sum(buffer_yes,axis=0) == 0:
            AP[index_of_query]=0
        else:
            AP[index_of_query]=np.sum(np.multiply(P,buffer_yes),axis=0)/np.sum(buffer_yes,axis=0)
        #print (index_of_query, AP[index_of_query])
        sum_tp = sum_tp + np.cumsum(buffer_yes)
        index_of_query+=1

    precision_at_k = np.divide(sum_tp,NS * QueryTimes)
    map_=np.mean(AP)
    print ('precision_at_k',precision_at_k)#array de valores
    save_obj(precision_at_k,'precision_at_k_n'+str(n)+'_k'+str(k))
    print ('map',map_)#valor numerico
    save_obj(map_,'map_n'+str(n)+'_k'+str(k))
    print ('n'+str(n)+'k'+str(k))

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
        print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N K[MODEL]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("K: Number of bits for the hash code (default: 12)")
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['k'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['model'] = sys.argv[3]
        main(**kwargs)