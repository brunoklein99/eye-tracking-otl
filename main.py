from os import listdir

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from data_load import get_mpii_datasets, get_custom_datasets
from model_vgg import NetVgg

device = torch.device('cuda')


def loss_fn(x_true, y_true, x_pred, y_pred):
    loss_x = torch.mean(torch.abs(x_true - x_pred))
    loss_y = torch.mean(torch.abs(y_true - y_pred))
    return loss_x + loss_y


def forward_backward_gen(model, loader):
    for x, x_true, y_true in loader:
        x = x.to(device)
        x_true = x_true.to(device)
        y_true = y_true.to(device)

        x_pred, y_pred = model(x)

        loss_batch = loss_fn(x_true, y_true, x_pred, y_pred)

        yield loss_batch


def evaluate(model, dataset, params):
    loader = DataLoader(dataset=dataset, batch_size=params['batch_size'])
    losses = []
    for loss in forward_backward_gen(model, loader):
        losses.append(float(loss))
    return np.mean(losses)


def train(model, parameters, train_dataset, valid_dataset, params):
    loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'])

    optimizer = SGD(parameters, lr=params['learning_rate'], momentum=0.9)

    epochs = params['epochs']
    for epoch in range(epochs):
        losses = []
        for i, loss in enumerate(forward_backward_gen(model, loader)):
            losses.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('train epoch {}/{} batch {}/{} loss {}'.format(epoch + 1, epochs, i + 1, len(loader), float(loss)))
        loss_train = np.mean(losses)
        loss_valid = evaluate(model, valid_dataset, params)
        print('epoch {} finished with train loss {} and valid loss {}'.format(epoch + 1, loss_train, loss_valid))


def create_or_load_model():
    weights = listdir('weights')
    if len(weights) > 1:
        weights_filename, *_ = sorted(weights, reverse=True)
        with open('weights/{}'.format(weights_filename), 'rb') as f:
            return torch.load(f)
    return NetVgg()


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset = get_mpii_datasets()

    params = {
        'learning_rate': 0.001,
        'epochs': 4,
        'batch_size': 8
    }

    model = create_or_load_model().to(device)

    train(model, model.parameters(), train_dataset, valid_dataset, params=params)

    test_loss = evaluate(model, test_dataset, params)

    print('test loss', test_loss)

    with open('weights/weights-{:2f}'.format(test_loss), 'wb') as f:
        torch.save(model, f)

    train_dataset, valid_dataset = get_custom_datasets()

    valid_loss_custom = evaluate(model, valid_dataset, params)
    print('valid loss custom', valid_loss_custom)

    params['batch_size'] = 1
    params['epochs'] = 1

    parameters = list(model.parameters())[16:]

    train(model, parameters, train_dataset, valid_dataset, params)
