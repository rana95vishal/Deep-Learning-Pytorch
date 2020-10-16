#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:40:36 2020

@author: vishalr
"""
CUDA_LAUNCH_BLOCKING="1"
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

#Data augmentation
transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# For trainning data
trainset = torchvision.datasets.CIFAR100(root='./data',
        train=True,download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=128, shuffle=True, num_workers=8)
# For testing data
testset = torchvision.datasets.CIFAR100(root='./data',
        train=False,download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
        batch_size=100, shuffle=False, num_workers=8)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(len(trainset.classes))
# Assuming that we are on a CUDA machine, this should print cuda

print(device)
#print(trainloader)



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
  
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

      
# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1], 2)
        self.layer3 = self.make_layer(block, 128, layers[2], 2)
        self.layer4 = self.make_layer(block, 256, layers[3], 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*4, 256)
        self.fc2 = nn.Linear(256, 200)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
      
      
net_args = {
    "block": ResidualBlock,
    "layers": [2, 4, 4, 2]
}

net = ResNet(**net_args)
#net = Net()
net.to(device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-4)
#scheduler = MultiStepLR(optimizer, milestones=[20,50], gamma=0.01)
#weight_decay=1e-2

for epoch in range(100):  
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels] and send it to gpu
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        running_loss += loss.item()
    print('[%d] Training loss: %.3f' %(epoch + 1, running_loss / i+1))
    #Test accuracy every 10 epochs
    if(epoch%1 == 0):
      running_loss = 0.0
      correct = 0
      total = 0
      with torch.no_grad():
          net.eval()
          #net.apply(apply_dropout)
          for data in testloader:
              inputs, labels = data[0].to(device), data[1].to(device)
              outputs = net(inputs)
              _, predicted = torch.max(outputs.data, 1)
              #print(predicted[1])
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

      print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))

print('Finished Training')