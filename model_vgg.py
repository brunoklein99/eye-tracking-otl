import math

import torch.nn as nn
from torchvision.models.vgg import make_layers, cfg


class NetVgg(nn.Module):

    def __init__(self):
        super().__init__()
        self.extractor = make_layers(cfg=cfg['A'], batch_norm=True)
        self.regressor = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2),
            nn.Sigmoid()
        )
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
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        x_axis = x[:, 0].view(-1, 1)
        y_axis = x[:, 1].view(-1, 1)
        return x_axis, y_axis
