# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.linear1 = nn.Linear(2, num_hid)
        self.linear2 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        r = torch.sqrt(torch.pow(input[:, 0], 2) + torch.pow(input[:, 1], 2))
        a = torch.atan2(input[:, 0], input[:, 1])
        input_co_ordinates = torch.stack((r, a), 1)
        output = self.linear1(input_co_ordinates)
        self.hid1 = self.tanh(output)
        output = self.linear2(self.hid1)
        output = self.sigmoid(output)
        # CHANGE CODE HERE
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.Linear1 = nn.Linear(2, num_hid)
        self.Linear2 = nn.Linear(num_hid, num_hid)
        self.Linear3 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.Linear1(input)
        self.hid1 = self.tanh(input)
        input = self.Linear2(self.hid1)
        self.hid2 = self.tanh(input)
        input = self.Linear3(self.hid2)
        output = self.sigmoid(input)
        # CHANGE CODE HERE
        return output

class MyRawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(MyRawNet, self).__init__()
        # INSERT CODE HERE
        self.Linear1 = nn.Linear(2, num_hid)
        self.Linear2 = nn.Linear(num_hid, num_hid)
        self.Linear3 = nn.Linear(num_hid, num_hid)
        self.Linear3 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.Linear1(input)
        self.hid1 = self.tanh(input)
        input = self.Linear2(self.hid1)
        self.hid2 = self.relu(input)
        input = self.Linear3(self.hid2)
        self.hid3 = self.relu(input)
        input = self.Linear4(self.input)
        output = self.sigmoid(input)
        # CHANGE CODE HERE
        return output


def graph_hidden(net, layer, node):
    #plt.clf()
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        net(grid)
        if layer == 1:
            pred = (net.hid1[:, node] >= 0).float()
        elif layer == 2:
            pred = (net.hid2[:, node] >= 0).float()
        else:
            raise ValueError("Please layer in [1, 2]")

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')

    # INSERT CODE HERE
