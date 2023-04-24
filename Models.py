import torch
import torch.nn as nn

class TwinHybridNetwork(nn.Module):
    def __init__(self, base_size=(384+512)):
        super(TwinHybridNetwork, self).__init__()
        self.feedforward1 = nn.Linear(base_size, 256)
        self.feedforward2 = nn.Linear(256, 512)
        #self.act = nn.LeakyReLU()
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input1, input2):
        x1 = self.act(self.feedforward1(input1))
        x1 = self.dropout(x1)
        x1 = self.act(self.feedforward2(x1))
        x2 = self.act(self.feedforward1(input2))
        x2 = self.dropout(x2)
        x2 = self.act(self.feedforward2(x2))
        return (x1, x2)

class TwinNetwork(nn.Module):
    def __init__(self):
        super(TwinNetwork, self).__init__()
        self.feedforward1 = nn.Linear(384, 100)
        self.feedforward2 = nn.Linear(100, 100)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input1, input2):
        x1 = self.act(self.feedforward1(input1))
        x1 = self.dropout(x1)
        x1 = self.act(self.feedforward2(x1))
        x2 = self.act(self.feedforward1(input2))
        x2 = self.dropout(x2)
        x2 = self.act(self.feedforward2(x2))
        return (x1, x2)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feedforward1 = nn.Linear(384, 50)
        self.feedforward2 = nn.Linear(50, 50)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.feedforward3 = nn.Linear(100, 1)

    def forward(self, input1, input2):
        x1 = self.act(self.feedforward1(input1))
        x1 = self.dropout(x1)
        x1 = self.act(self.feedforward2(x1))
        x2 = self.act(self.feedforward1(input2))
        x2 = self.dropout(x2)
        x2 = self.act(self.feedforward2(x2))
        x3 = torch.cat((x1, x2), 1)
        res = self.act(self.feedforward3(x3))
        return (res)
