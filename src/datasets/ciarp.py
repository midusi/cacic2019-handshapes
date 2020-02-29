"""Load ciarp dataset"""

import os
import numpy as np
import handshape_datasets as hd
from pathlib import Path

def load_ciarp(dataset_name):
    """
    Load ciarp dataset.

    Returns (x, y): as dataset x and y.

    """
    dataset_path = '/tf/data/{}/data'.format(dataset_name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    data = hd.load(dataset_name, Path(dataset_path))

    # TODO: define best way to do this

    x_train, y_train = data['train_Kinect_WithoutGabor']
    x_test, y_test = data['test_Kinect_WithoutGabor']

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    return x, y
