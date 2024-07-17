import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.linear = nn.Linear(7 * 7 * 64, 10)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight.data)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.functional.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.pool2(out)

        out = out.view(out.size(0), -1)

        return self.linear(out)

