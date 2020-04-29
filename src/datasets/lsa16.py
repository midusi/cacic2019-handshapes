"""Load lsa16 dataset"""

import os
import handshape_datasets as hd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.model_selection import train_test_split_balanced

def load_lsa16(dataset_name, path, train_size=None, test_size=None, n_train_per_class=0, n_test_per_class=0):
    """
    Load lsa16 dataset.

    Returns (x, y): as dataset x and y.

    """

    if path == None:
        path = '/tf/data/{}/data'.format(dataset_name)

    if not os.path.exists(path):
        os.makedirs(path)
    data = hd.load(dataset_name, Path(path))

    x, y = data[0], data[1]['y']

    split = train_test_split if n_train_per_class <= 0 else train_test_split_balanced

    if n_train_per_class <= 0:
        x_train, x_test, y_train, y_test = split(x, y, train_size=train_size, test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = split(x, y, n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)

    return (x_train, y_train), (x_test, y_test)
