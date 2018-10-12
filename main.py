from os import listdir

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from data_load import load_mpii_dataframes, get_dataset
from model_vgg import NetVgg


def loss_fn(x_true, y_true, x_pred, y_pred):
    loss_x = torch.mean(torch.abs(x_true - x_pred))
    loss_y = torch.mean(torch.abs(y_true - y_pred))
    return loss_x + loss_y


def evaluate(model, data_frame, params, device):
    dataset = get_dataset(data_frame)

    loader = DataLoader(dataset=dataset, batch_size=params['batch_size'])

    losses = []
    for i, (x, m, x_true, y_true) in enumerate(loader):
        x = x.to(device)
        # m = m.to(device)
        x_true = x_true.to(device)
        y_true = y_true.to(device)

        x_pred, y_pred = model(x)

        loss_batch = loss_fn(x_true, y_true, x_pred, y_pred)

        losses.append(float(loss_batch))

    return np.mean(losses)


def train(model, train_frame, valid_frame, params, device):
    dataset = get_dataset(train_frame)

    loader = DataLoader(dataset=dataset, batch_size=params['batch_size'])

    optimizer = SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)

    epochs = params['epochs']
    for epoch in range(epochs):
        losses = []
        for i, (x, m, x_true, y_true) in enumerate(loader):
            x = x.to(device)
            # m = m.to(device)
            x_true = x_true.to(device)
            y_true = y_true.to(device)

            x_pred, y_pred = model(x)

            loss_batch = loss_fn(x_true, y_true, x_pred, y_pred)

            losses.append(float(loss_batch))

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            if i % 50 == 0:
                print('train epoch {}/{} batch {}/{} loss {}'.format(epoch + 1, epochs, i + 1, len(loader),
                                                                     float(loss_batch)))
        loss_train = np.mean(losses)
        loss_valid = evaluate(model, valid_frame, params, device)
        print('epoch {} finished with train loss {} and valid loss {}'.format(epoch + 1, loss_train, loss_valid))

    return model


if __name__ == '__main__':
    df_train, df_valid, df_test = load_mpii_dataframes()

    params = {
        'learning_rate': 0.001,
        'epochs': 4,
        'batch_size': 8
    }

    device = torch.device('cuda')

    weights = listdir('weights')
    if len(weights) > 1:
        weights_filename, *_ = sorted(weights, reverse=True)
        with open('weights/{}'.format(weights_filename), 'rb') as f:
            model = torch.load(f)
    else:
        model = NetVgg().to(device)

    model = train(model, train_frame=df_train, valid_frame=df_valid, params=params, device=device)

    test_loss = evaluate(model, df_test, params, device)

    print('test loss', test_loss)

    with open('weights/weights-{:2f}'.format(test_loss), 'wb') as f:
        torch.save(model, f)
