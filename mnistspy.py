#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:34:12 2019

@author: vishalr
"""

# -*- coding: utf-8 -*-

#implementation of a simple feedforward network from scratch
import numpy as np
import h5py
import copy
from random import randint
import matplotlib.pyplot as plt

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
#number of units in the hidden layer
num_nodes = 100

#model parameters for a single layer network
model = {}
model['W1'] = np.random.randn(num_nodes,num_inputs) / np.sqrt(num_nodes)
model['b1'] = np.reshape(np.random.randn(num_nodes) / np.sqrt(num_nodes),(-1,1))
model['C'] = np.random.randn(num_outputs,num_nodes) / np.sqrt(num_nodes)
model['b2'] = np.reshape(np.random.randn(num_outputs) / np.sqrt(num_nodes),(-1,1))
model_grads = copy.deepcopy(model)

#activation function
def relu_function(z):
    ZZ = np.maximum(z,0)
    return ZZ

#derivative of activation function
def relu_der(z):
    del_z = [0 if i<0 else 1 for i in z]
    return del_z

#implements Z = WX+b1
def forward(x,y, model):
    ztemp = np.reshape(np.dot(model['W1'], x),(-1,1))
    Z = np.add(ztemp,model['b1'])
    return Z

#implement fx = CH+b2
def combine_out(H,model):
    U = np.dot(model['C'],H) + model['b2']
    fx = relu_function(U)
    return fx

#backpropagation
def partial_der(fx,y_oh,x,H,z,model,LR):
    del_U =  np.subtract(fx, y_oh) 
    del_U = np.reshape(del_U, (-1, 1))
    H = np.reshape(H, (-1, 1))
    del_b2 = del_U
    del_C = np.dot(del_U,np.transpose(H))
    delta = np.dot(np.transpose(model['C']),del_U)
    del_z = np.reshape(relu_der(z),(-1,1))
    del_b1 = delta*del_z
    x = np.reshape(x,(-1,1))
    del_W1 = np.dot(del_b1,np.transpose(x))
    model['C'] = model['C'] - LR*del_C
    model['b2'] = model['b2'] - LR*del_b2
    model['b1'] = model['b1'] - LR*del_b1
    model['W1'] = model['W1'] - LR*del_W1
    
LR = 0.01
num_epochs = 25
train_err = []
test_err = []
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.01
    if (epochs > 10):
        LR = 0.001
    if (epochs > 15):
        LR = 0.0001
    total_correct = 0
    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z = forward(x, y, model)
        H = relu_function(Z)
        #converting y to one hot vector i.e vector e(h)
        y_oh = np.reshape(np.zeros(num_outputs),(-1,1))
        y_oh[y] = 1
        fx = combine_out(H,model)
        loss = -np.sum(y_oh*np.log(fx+1e-9))
        prediction = np.argmax(fx)
        if (prediction == y):
            total_correct += 1
        partial_der(fx,y_oh,x,H,Z,model,LR)
    train_err.append(total_correct/np.float(len(x_train) ))
    
    total_correct = 0
    for n in range( len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        p = forward(x, y, model)
        pz = relu_function(p)
        pout = combine_out(pz,model)
        prediction = np.argmax(pout)
        if (prediction == y):
            total_correct += 1
    test_err.append(total_correct/np.float(len(x_test) ) )

plt.plot(np.arange(1,num_epochs+1),train_err, label = 'training error')
plt.plot(np.arange(1,num_epochs+1),test_err, label = 'test error')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
#end