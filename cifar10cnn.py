#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 19:53:40 2020

@author: vishalr
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat',
#          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=2)
        self.drop2 = nn.Dropout(p=0.5)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=2)
        self.drop4 = nn.Dropout(p=0.5)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=2)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.drop6 = nn.Dropout(p=0.5)
        
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.bn8 = nn.BatchNorm2d(64)
        self.drop8 = nn.Dropout(p=0.5)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop6(F.relu(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.drop8(F.relu(self.bn8(self.conv8(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)

learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[5,15], gamma=0.1)
#weight_decay=1e-2
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

print('Finished Training')