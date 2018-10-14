import torch.utils.data as data


class MpiiNormalizingDataset(data.Dataset):

    def __init__(self, dataset, mean, stddev):
        self.dataset = dataset
        self.mean = mean
        self.stddev = stddev

    def __getitem__(self, i):
        x, x_axis, y_axis = self.dataset[i]
        x /= 255.
        x[:, :, 0] -= self.mean[0]
        x[:, :, 1] -= self.mean[1]
        x[:, :, 2] -= self.mean[2]
        x[:, :, 0] /= self.stddev[0]
        x[:, :, 1] /= self.stddev[1]
        x[:, :, 2] /= self.stddev[2]
        return x, x_axis, y_axis

    def __len__(self):
        return len(self.dataset)
