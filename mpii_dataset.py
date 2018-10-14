import cv2
import numpy as np
import torch.utils.data as data


class MpiiDataset(data.Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = cv2.imread(r['imagename']).astype(dtype=np.float32)
        x_axis = np.array(r['x'], ndmin=1)
        y_axis = np.array(r['y'], ndmin=1)
        return x, x_axis, y_axis

    def __len__(self):
        return len(self.df)
