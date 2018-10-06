import torch
import torch.utils.data as data


class TensorDataset(data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        t = self.dataset[i]
        return TensorDataset.map(t, TensorDataset.to_tensor)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def to_tensor(x):
        return torch.from_numpy(x).float()

    @staticmethod
    def map(t, f):
        r = ()
        for e in t:
            r += (f(e),)
        return r
