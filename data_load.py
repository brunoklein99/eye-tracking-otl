import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mpii_dataset import MpiiDataset
from mpii_normalizing_dataset import MpiiNormalizingDataset
from resize_dataset import ResizeDataset
from tensor_dataset import TensorDataset


def load_mpii_dataframes():
    df = pd.read_csv('data/mpii_prepared/metadata.csv')
    df = df.sample(frac=1, random_state=1)
    df_train_valid, df_test = train_test_split(df, random_state=1, test_size=0.15)
    df_train, df_valid = train_test_split(df_train_valid, random_state=1, test_size=0.1765)
    return df_train, df_valid, df_test


def load_custom_dataframes(n, split_index, shuffle=False):
    df = pd.read_csv('data/custom{}_prepared/metadata.csv'.format(n))
    if shuffle:
        df = df.sample(frac=1)
    df_train = df[:split_index]
    df_valid = df[split_index:]
    return df_train, df_valid


def decorate_dataset(dataset):
    dataset = ResizeDataset(dataset)
    dataset = MpiiNormalizingDataset(dataset,
                                     mean=(0.33320256,
                                           0.35879958,
                                           0.45563497),
                                     stddev=(np.sqrt(0.05785664),
                                             np.sqrt(0.06049888),
                                             np.sqrt(0.07370879)))
    dataset = TensorDataset(dataset)
    return dataset


def get_mpii_datasets():
    df_train, df_valid, df_test = load_mpii_dataframes()
    train_dataset = decorate_dataset(MpiiDataset(df_train))
    valid_dataset = decorate_dataset(MpiiDataset(df_valid))
    test_dataset = decorate_dataset(MpiiDataset(df_test))
    return train_dataset, valid_dataset, test_dataset


def get_custom_datasets(n, split_index, shuffle=False):
    df_train, df_valid = load_custom_dataframes(n=n, split_index=split_index, shuffle=shuffle)
    train_dataset = decorate_dataset(MpiiDataset(df_train))
    valid_dataset = decorate_dataset(MpiiDataset(df_valid))
    return train_dataset, valid_dataset
