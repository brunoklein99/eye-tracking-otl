import cv2
import numpy as np
import torch
import torch.utils.data as data


class MpiiDataset(data.Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = torch.from_numpy(cv2.imread(r['imagename'])).float()
        m = torch.from_numpy(cv2.imread(r['maskname'])).float()
        x_axis = torch.from_numpy(np.array(r['x'], ndmin=1))
        y_axis = torch.from_numpy(np.array(r['y'], ndmin=1))
        return x, m, x_axis, y_axis

    def __len__(self):
        return len(self.df)
