from math import pi

import torch

import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

import settings
from data_load import load_mpii_dataframes
from device_dataset import DeviceDataset
from model import Net
from mpii_dataset import MpiiDataset
from mpii_normalizing_dataset import MpiiNormalizingDataset
from tensor_dataset import TensorDataset


def loss_func(z, mean, var):
    t1 = 2 * pi * var
    t1 = torch.pow(t1, -1 / 2)
    t1 = torch.log(t1)

    t2 = z - mean
    t2 = torch.pow(t2, 2)
    t2 = - t2
    t2 = t2 / (2 * var)

    loss = t1 + t2
    # loss = torch.exp(loss)
    # loss = torch.log(loss)
    loss = torch.mean(loss)
    loss = -loss

    return loss


def evaluate(model, x, x_axis, y_axis):
    mean_x, var_x, mean_y, var_y = model(x)

    x_axis = x_axis.cpu().detach().numpy()
    y_axis = y_axis.cpu().detach().numpy()

    mean_x = mean_x.cpu().detach().numpy()
    mean_y = mean_y.cpu().detach().numpy()
    var_x = var_x.cpu().detach().numpy()
    var_y = var_y.cpu().detach().numpy()

    x_pred = np.zeros_like(mean_x)
    y_pred = np.zeros_like(mean_y)

    n_samples = 20
    for i in range(n_samples):
        x_pred[:] += np.random.normal(mean_x, np.power(var_x, 2))
        y_pred[:] += np.random.normal(mean_y, np.power(var_y, 2))
    x_pred /= n_samples
    y_pred /= n_samples

    diff_x = np.mean(np.abs(x_axis - x_pred))
    diff_y = np.mean(np.abs(y_axis - y_pred))

    return diff_x, diff_y


def train(data_frame, params):
    device = torch.device('cuda')

    dataset = MpiiDataset(data_frame)
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

    model = Net().to(device)

    optimizer = Adam(model.parameters(), lr=params['learning_rate'])

    epochs = params['epochs']
    for epoch in range(epochs):
        losses = []
        for i, (x, m, x_axis, y_axis) in enumerate(loader):
            mean_x, var_x, mean_y, var_y = model(x)
            x_loss = loss_func(x_axis, mean=mean_x, var=var_x)
            y_loss = loss_func(y_axis, mean=mean_y, var=var_y)
            loss = x_loss + y_loss

            losses.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('training epoch {}/{} batch {}/{} loss {}'.format(epoch + 1, epochs, i + 1, len(loader),
                                                                        float(loss)))
                diff_x, diff_y = evaluate(model, x, x_axis, y_axis)

                print('diff_x', diff_x)
                print('diff_y', diff_y)


if __name__ == '__main__':
    df_train, df_valid, df_test = load_mpii_dataframes()
    train(df_train, params={
        'learning_rate': 1e-3,
        'epochs': 5
    })
