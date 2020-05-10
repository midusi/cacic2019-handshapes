"""Load lsa16 dataset"""

import os
import numpy as np
import handshape_datasets as hd
from pathlib import Path
from src.utils.model_selection import train_test_split_balanced
from .util import load_from_split

def load_lsa16(config, path=None):
    """
    Load lsa16 dataset.

    Returns (x, y): as dataset x and y.

    """

    train_size=config['data.train_size']
    test_size=config['data.test_size']
    n_train_per_class=config['data.n_train_per_class']
    n_test_per_class=config['data.n_test_per_class']

    if path == None:
        path = '/tf/data/{}'.format(config['data.dataset'])
        data_dir = os.path.join(path, 'data')

    if not os.path.exists(path):
        os.makedirs(path)

    data = hd.load(config['data.dataset'], Path(data_dir))

    if config['data.split']:
        split_dir = os.path.join(path, 'splits', config['data.split'])
        split_file = lambda split: os.path.join(split_dir, f"{split}.txt")

        x_train, y_train = load_from_split(config['data.dataset'], config['data.version'], data_dir, split_file('train'))
        x_test, y_test = load_from_split(config['data.dataset'], config['data.version'], data_dir, split_file('test'))
        x_val, y_val = load_from_split(config['data.dataset'], config['data.version'], data_dir, split_file('val'))
    else:
        x, y = data[0], data[1]['y']

        n_train_per_class = int(np.round(n_train_per_class * 1.6))
        x_train, x_test, y_train, y_test = train_test_split_balanced(np.array(x), np.array(y), train_size=train_size, test_size=test_size,
                                                n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)
        x_train, x_val, y_train, y_val = train_test_split_balanced(x_train, y_train, train_size=0.8, n_train_per_class=n_train_per_class, test_size=0.2)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
