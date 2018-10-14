import torch.utils.data as data
import cv2


class ResizeDataset(data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        x, x_axis, y_axis = self.dataset[i]
        x = cv2.resize(x, dsize=(255, 255))
        return x, x_axis, y_axis

    def __len__(self):
        return len(self.dataset)
