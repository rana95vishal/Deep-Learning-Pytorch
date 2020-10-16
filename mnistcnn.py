#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:24:29 2019

@author: vishalr
"""
#implementation of a simple CNN from scratch
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

#reshape image vector into 28*28 images
x_train = np.reshape(x_train,(60000,28,28))
x_test = np.reshape(x_test,(10000,28,28))

#number of inputs
dx = 28
dy = 28
#number of outputs
num_outputs = 10
#number of units in the hidden layer
num_nodes = 100
#number of channels
channel = 3
#shape of filters
kx = 5
ky = 5

#model parameters
model = {}
model['K'] = np.random.randn(channel,kx,ky) / np.sqrt(kx*ky)
model['W'] = np.random.randn(num_outputs,(dx-kx+1)*(dy-ky+1),channel) / np.sqrt((dx-kx+1)* (dy-ky+1))
model['b'] = np.reshape(np.random.randn(num_outputs) / np.sqrt(num_outputs),(-1,1))
model_grads = copy.deepcopy(model)

#extract windows for performing convolution
def extract_window(x):
    x_shape = np.shape(x)
    x_temp = np.zeros((x_shape[0],(dx-kx+1)*(dy-ky+1),kx,ky))
    for n in range(x_shape[0]):
        for i in range(dx-kx+1):
            for j in range(dy-ky+1):
                x_temp[n][i*(dx-kx+1)+j] = x[n,i:i+kx,j:j+ky]
    return(x_temp)
     
#extracting windows for backpropagation step       
def extract_window_b(x):
    x_shape = np.shape(x)
    x_temp = np.zeros((x_shape[0],kx*ky,dx-kx+1,dy-ky+1))
    for n in range(x_shape[0]):
        for i in range(kx):
            for j in range(ky):
                x_temp[n][i*(kx)+j] = x[n,i:i+dx-kx+1,j:j+dy-ky+1]
    return(x_temp)
    
x_train_b = extract_window_b(x_train)
x_train = extract_window(x_train)
x_test = extract_window(x_test)
    

#convolution function
def convol_b(x,delta):
    x_shape = np.shape(x)
    Z=np.zeros((x_shape[0],channel))
    for i in range(x_shape[0]):
        Z[i][:] = np.sum(np.sum(np.multiply(x[i],delta),axis=2),axis=1)
    return(Z)
    
def relu_function(z):
    ZZ = np.maximum(z,0)
    return ZZ

#derivative of activation function
def relu_der(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z


def combine_out(H,model):
    Utemp = np.reshape(np.tensordot(model['W'],H,axes=([1,2],[0,1])),(-1,1))
    U = np.add(Utemp,model['b'])
    return U

#backpropagation
def partial_der(fx,e_h,x,H,z,model,LR,n_random):
    del_U =  np.subtract(fx, e_h) 
    del_U = np.reshape(del_U, (-1, 1))
    del_W = np.zeros((num_outputs,(dx-kx+1)*(dy-ky+1),channel))
    for i in range(num_outputs):
        del_W[i] = del_U[i]*H   
    delta = np.tensordot(model['W'],del_U,axes=([0],[0]))
    del_z = np.transpose(relu_der(z)*delta[:,:,0])
    del_zr = np.zeros((3,(dx-kx+1),(dy-ky+1)))
    for i in range(channel):
        del_zr[i] =  np.reshape(del_z[i],((dx-kx+1),(dy-ky+1)))
    del_K = (convol_b(x_train_b[n_random],del_zr))
    del_Kr = np.zeros((3,kx,ky))
    for i in range(channel):
        del_Kr[i] = np.reshape(del_K[:,i],(kx,ky))
    model['b'] = model['b'] - LR*del_U
    model['K'] = model['K'] - LR*del_Kr
    model['W'] = model['W'] - LR*del_W
    

LR = 0.01
num_epochs = 20
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
        Z = convol_b(x, model['K'])
        H = relu_function(Z)
        fx = combine_out(H,model)
        prediction = np.argmax(fx)
        e_h = np.reshape(np.zeros(num_outputs),(-1,1))
        e_h[y] = 1
        #converting y to one hot vector i.e vector e(h)
        #loss = -np.sum(y_oh*np.log(fx+1e-9))
        prediction = np.argmax(fx)
        if (prediction == y):
            total_correct += 1
        partial_der(fx,e_h,x,H,Z,model,LR,n_random)
    train_err.append(total_correct/np.float(len(x_train)))
    total_correct = 0
    for n in range( len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        #Z = convol(x, model)
        Z = convol_b(x,model['K'])
        H = relu_function(Z)
        fx = combine_out(H,model)
        prediction = np.argmax(fx)
        if (prediction == y):
            total_correct += 1
    test_err.append(total_correct/np.float(len(x_test) ) )


total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    Z = convol_b(x, model['K'])
    H = relu_function(Z)
    fx = combine_out(H,model)
    prediction = np.argmax(fx)
    if (prediction == y):
        total_correct += 1
print("Test accuracy", total_correct/np.float(len(x_test) ) )

plt.plot(np.arange(1,num_epochs+1),train_err, label = 'training error')
plt.plot(np.arange(1,num_epochs+1),test_err, label = 'test error')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()