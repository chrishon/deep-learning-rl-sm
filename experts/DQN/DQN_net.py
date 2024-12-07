import torch
import torch.nn as nn


class DQN_net(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN_net, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.dropOut = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        NumAdditionalArgsLinLayer = 35
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer)
        self.h0 = nn.Linear(linear_input_size, 100)
        self.h1 = nn.Linear(100 + NumAdditionalArgsLinLayer, 100)
        self.h2 = nn.Linear(100, 100)
        self.headPlanner = nn.Linear(100, outputs)

    def forward(self, state, state_additional):
        state_out = torch.relu(self.h0(torch.relu(self.flatten(self.dropOut(self.conv2d(state))))))
        x_last_layer = torch.concat((state_out, state_additional), dim=1)
        return torch.softmax(self.headPlanner(torch.relu(self.h2(torch.relu(self.h1(x_last_layer))))), dim=1)
