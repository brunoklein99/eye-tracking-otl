import torch.nn as nn
import torch.nn.functional as F


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

        self.linear1 = nn.Linear(in_features=36864, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)

        self.linear_mean_x = nn.Linear(in_features=4096, out_features=1)
        self.linear_var_x = nn.Linear(in_features=4096, out_features=1)

        self.linear_mean_y = nn.Linear(in_features=4096, out_features=1)
        self.linear_var_y = nn.Linear(in_features=4096, out_features=1)

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

        features = v.view(x.size(0), -1)

        out = F.relu(self.linear1(features))
        out = F.relu(self.linear2(out))

        mean_x = F.softplus(self.linear_mean_x(out))
        var_x = F.softplus(self.linear_var_x(out))

        mean_y = F.softplus(self.linear_mean_y(out))
        var_y = F.softplus(self.linear_var_y(out))

        return mean_x, var_x, mean_y, var_y
