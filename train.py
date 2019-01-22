from argparse import ArgumentParser
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch.optim import SGD
from torch.utils.data import DataLoader

from data_load import get_mpii_datasets, get_custom_datasets
from model_vgg import NetVgg

device = torch.device('cuda')
screen_width = 1920
screen_height = 1080


def loss_fn(x_true, y_true, x_pred, y_pred):
    return torch.mean(torch.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2))


def plot_and_save_debug_image(i, face_crop, x_true, y_true, x_pred, y_pred, loss):
    img = np.zeros(shape=(screen_height, screen_width, 3))
    face_crop *= 255
    face_crop = face_crop.astype(np.uint8)

    # picture in picture bottom right margin 10
    img[815:1070, 1655:1910, :] = face_crop
    x_true = int(x_true * screen_width)
    x_pred = int(x_pred * screen_width)
    y_true = int(y_true * screen_height)
    y_pred = int(y_pred * screen_height)
    img = cv2.circle(img, center=(x_true, y_true), radius=100, color=(0, 255, 0), thickness=2)
    img = cv2.circle(img, center=(x_pred, y_pred), radius=100, color=(255, 0, 0), thickness=2)
    img = cv2.putText(img, '{:.2f}'.format(loss), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2, cv2.LINE_AA)
    cv2.imwrite('simulation/{}.jpg'.format(i), img)
    print()


def forward_backward_gen(model, loader):
    for i, (x, x_true, y_true) in enumerate(loader):
        x = x.to(device)
        x_true = x_true.to(device)
        y_true = y_true.to(device)

        x_pred, y_pred = model(x)

        loss_batch = loss_fn(x_true, y_true, x_pred, y_pred)

        if args.video:
            plot_and_save_debug_image(
                i,
                np.squeeze(x.cpu().data.numpy()),
                float(x_true.cpu().data.numpy()),
                float(y_true.cpu().data.numpy()),
                float(x_pred.cpu().data.numpy()),
                float(y_pred.cpu().data.numpy()),
                float(loss_batch.cpu().data.numpy())
            )

        yield loss_batch


def evaluate(model, dataset, params):
    loader = DataLoader(dataset=dataset, batch_size=params['batch_size'])
    losses = []
    for loss in forward_backward_gen(model, loader):
        losses.append(float(loss))
    return np.mean(losses)


def train(model, parameters, train_dataset, valid_dataset, params, threshold_loss=None):
    loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'])

    optimizer = SGD(parameters, lr=params['learning_rate'], momentum=0.9)

    epochs = params['epochs']

    loss_train_all = []
    loss_valid_all = []
    for epoch in range(epochs):
        losses = []
        for i, loss in enumerate(forward_backward_gen(model, loader)):
            losses.append(float(loss))

            if threshold_loss is None or loss > threshold_loss:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % 50 == 0:
                print('train epoch {}/{} batch {}/{} loss {}'.format(
                    epoch + 1, epochs, i + 1, len(loader), float(loss)
                ))
        loss_train = np.mean(losses)
        loss_valid = evaluate(model, valid_dataset, params)
        loss_train_all.append(loss_train)
        loss_valid_all.append(loss_valid)
        print('epoch {} finished with train loss {} and valid loss {}'.format(epoch + 1, loss_train, loss_valid))
    return loss_train_all, loss_valid_all


def create_or_load_model():
    weights = listdir('weights')
    if len(weights) > 1:
        weights_filename, *_ = sorted(weights, reverse=True)
        loss = float(weights_filename.split('-')[1])
        with open('weights/{}'.format(weights_filename), 'rb') as f:
            return torch.load(f), loss
    return NetVgg(), None


def fine_tune_stage(model, params, n, threshold_loss, split_index):
    print('stage {}'.format(n))

    params['batch_size'] = 1
    params['epochs'] = 1

    train_dataset, valid_dataset = get_custom_datasets(n, split_index)

    valid_loss_custom = evaluate(model, valid_dataset, params)
    print('valid loss custom', valid_loss_custom)

    parameters = list(model.parameters())[16:]

    train(model, parameters, train_dataset, valid_dataset, params, threshold_loss)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='generate video frames', action='store_true')

    args = parser.parse_args()

    train_dataset, valid_dataset, test_dataset = get_mpii_datasets()

    params = {
        'learning_rate': 0.001,
        'epochs': 4,
        'batch_size': 8
    }

    model, test_loss = create_or_load_model()

    print(model)

    model = model.to(device)
    if test_loss is None:
        loss_train_all, loss_valid_all = train(model, model.parameters(), train_dataset, valid_dataset, params=params)

        plt.plot(range(len(loss_train_all)), loss_train_all, label='train')
        plt.plot(range(len(loss_valid_all)), loss_valid_all, label='valid')
        plt.legend()
        plt.show()

        test_loss = evaluate(model, test_dataset, params)

        print('test loss', test_loss)

        with open('weights/weights-{:2f}'.format(test_loss), 'wb') as f:
            torch.save(model, f)

    fine_tune_stage(model, params, n=1, threshold_loss=test_loss, split_index=400)
    fine_tune_stage(model, params, n=2, threshold_loss=test_loss, split_index=400)
    fine_tune_stage(model, params, n=3, threshold_loss=test_loss, split_index=100)
