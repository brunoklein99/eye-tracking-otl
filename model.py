import math

import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.maxp2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        self.maxp5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv1_spatial = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv2_spatial = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv3_spatial = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)

        self.linear1 = nn.Linear(in_features=12 * 12 * 256, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        u = F.relu(self.conv1(x))
        u = self.maxp1(u)

        u = F.relu(self.conv2(u))
        u = self.maxp2(u)

        u = F.relu(self.conv3(u))
        u = F.relu(self.conv4(u))
        u = F.relu(self.conv5(u))
        u = self.maxp5(u)

        w = F.relu(self.conv1_spatial(u))
        w = F.relu(self.conv2_spatial(w))
        w = F.relu(self.conv3_spatial(w))

        v = u * w

        f = v.view(x.size(0), -1)

        out = F.relu(self.linear1(f))
        out = F.relu(self.linear2(out))
        out = torch.sigmoid(self.linear3(out))
        x_axis = out[:, 0].view(-1, 1)
        y_axis = out[:, 1].view(-1, 1)
        return x_axis, y_axis
