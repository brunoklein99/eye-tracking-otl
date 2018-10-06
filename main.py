import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

import settings
import torch.nn.functional as F
from data_load import load_mpii_dataframes
from device_dataset import DeviceDataset
from model import Net
from model_vgg import NetVgg
from mpii_dataset import MpiiDataset
from mpii_normalizing_dataset import MpiiNormalizingDataset
from resize_dataset import ResizeDataset
from tensor_dataset import TensorDataset


def train(data_frame, params):
    device = torch.device('cuda')

    dataset = MpiiDataset(data_frame)
    dataset = ResizeDataset(dataset)
    dataset = MpiiNormalizingDataset(dataset,
                                     mean=(0.33320256,
                                           0.35879958,
                                           0.45563497),
                                     stddev=(np.sqrt(0.05785664),
                                             np.sqrt(0.06049888),
                                             np.sqrt(0.07370879)))
    dataset = TensorDataset(dataset)
    dataset = DeviceDataset(dataset, device)

    loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE)

    model = NetVgg().to(device)

    optimizer = SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)

    epochs = params['epochs']
    for epoch in range(epochs):
        losses = []
        for i, (x, m, x_true, y_true) in enumerate(loader):
            x_pred, y_pred = model(x)

            # loss = F.mse_loss(x_pred, x_true) + F.mse_loss(y_pred, y_true)
            # loss = torch.mean(loss)
            loss_x = torch.mean(torch.abs(x_true - x_pred))
            loss_y = torch.mean(torch.abs(y_true - y_pred))
            loss = loss_x + loss_y

            losses.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('training epoch {}/{} batch {}/{} loss {}'.format(epoch + 1, epochs, i + 1, len(loader), float(loss)))


if __name__ == '__main__':
    df_train, df_valid, df_test = load_mpii_dataframes()
    train(df_train, params={
        'learning_rate': 0.001,
        'epochs': 999999999999
    })
