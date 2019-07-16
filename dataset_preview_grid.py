from os import listdir
from os.path import join
from random import choice

import cv2
import numpy as np


def create_array(rows, cols):
    return np.zeros(shape=(rows * 100, cols * 100, 3), dtype=np.uint8)


def set_image(arr, row, col, img):
    l_row = row * 100
    h_row = l_row + 100
    l_col = col * 100
    h_col = l_col + 100
    arr[l_row:h_row, l_col:h_col] = img


def plot_mpii_gaze():
    candidate_images = [x for x in listdir('data/mpii_prepared') if x.endswith('.jpg')]

    arr = create_array(5, 5)

    for i in range(5):
        for j in range(5):
            set_image(arr, i, j,
                      cv2.resize(cv2.imread('data/mpii_prepared/{}'.format(choice(candidate_images)), cv2.IMREAD_COLOR),
                                 dsize=(100, 100)))

    # cv2.imshow('', arr)
    # cv2.waitKey()
    cv2.imwrite('images/mpii-preview.jpg', arr)


def plot_custom():
    candidate_images = [join('data/custom1_prepared', x) for x in listdir('data/custom1_prepared') if x.endswith('.jpg')] + \
                       [join('data/custom2_prepared', x) for x in listdir('data/custom2_prepared') if x.endswith('.jpg')]

    arr = create_array(5, 5)

    for i in range(5):
        for j in range(5):
            set_image(arr, i, j, cv2.resize(cv2.imread(choice(candidate_images), cv2.IMREAD_COLOR), dsize=(100, 100)))

    # cv2.imshow('', arr)
    # cv2.waitKey()
    cv2.imwrite('images/custom-preview.jpg', arr)


if __name__ == '__main__':
    # plot_mpii_gaze()
    plot_custom()
