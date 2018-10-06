import torch.utils.data as data


class DeviceDataset(data.Dataset):

    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

    def __getitem__(self, i):
        t = self.dataset[i]
        return self.map(t)

    def __len__(self):
        return len(self.dataset)

    def map(self, t):
        r = ()
        for e in t:
            r += (e.to(self.device),)
        return r
