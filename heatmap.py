from argparse import ArgumentParser

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', type=int, dest='index', help='index of environment dataset')
    args = parser.parse_args()

    df = pd.read_csv('data/custom{}_prepared/metadata.csv'.format(args.index))
    x = df['x'].values
    y = df['y'].values

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
