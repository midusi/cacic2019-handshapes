"""Load lsa16 dataset"""

import os
import handshape_datasets as hd
from pathlib import Path

def load_lsa16(dataset_name):
    """
    Load lsa16 dataset.

    Returns (x, y): as dataset x and y.

    """
    dataset_path = '/tf/data/{}/data'.format(dataset_name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    data = hd.load(dataset_name, Path(dataset_path))

    return data[0], data[1]['y']