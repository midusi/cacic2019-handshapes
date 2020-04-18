"""Load Ciarp dataset"""

import os
import numpy as np
import handshape_datasets as hd
from pathlib import Path

def load_ciarp(dataset_name):
    """
    Load Ciarp dataset.

    Returns (x, y): as dataset x and y.

    """
    dataset_path = '/tf/data/{}/data'.format(dataset_name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    X, meta = hd.load(dataset_name, Path(dataset_path), version='WithoutGabor')

    return X, meta['y']
