# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = torch.nn.Linear(28 * 28, 10)
        self.softmax = torch.nn.LogSoftmax()


    def forward(self, x):
        batch = x.size()[0]
        x = x.view(batch, -1)
        x = self.linear(x)
        x = self.softmax(x)
        return x # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.linear1 = torch.nn.Linear(28*28, 256)
        self.linear2 = torch.nn.Linear(256, 10)
        self.softmax = torch.nn.LogSoftmax()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        batch = x.size()[0]
        x = x.view(batch, -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = torch.nn.Conv2d(1, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.pool = torch.nn.MaxPool2d(3, 2)
        self.relu = torch.nn.ReLU()

        self.linear = torch.nn.Linear(128*10*10, 10)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        batch = x.size()[0]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(batch, -1)
        x = self.linear(x)
        x = self.softmax(x)

        return x# CHANGE CODE HERE



class MyNetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(MyNetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = torch.nn.Conv2d(1, 64, 3)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.pool = torch.nn.MaxPool2d(3, 2)
        self.relu = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(128*10*10, 64*10*10)
        self.dropout = nn.Dropout(0.7)
        self.linear2 = torch.nn.Linear(64*10*10, 10)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        batch = x.size()[0]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(batch, -1)
        x = self.linear1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x# CHANGE CODE HERE
