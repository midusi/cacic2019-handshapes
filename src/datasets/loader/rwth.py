"""Load rwth dataset"""

import os
import numpy as np
import handshape_datasets as hd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.model_selection import train_test_split_balanced
from .util import load_from_split

def load_rwth(config, path=None):
    """
    Load rwth dataset.

    Returns (x, y): as dataset x and y.

    """

    train_size=config['data.train_size']
    test_size=config['data.test_size']
    n_train_per_class=config['data.n_train_per_class']
    n_test_per_class=config['data.n_test_per_class']

    if path == None:
        path = '/tf/data/{}'.format(config['data.dataset'])
        data_dir = os.path.join(path, 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = hd.load(config['data.dataset'], Path(data_dir))

    if config['data.split']:
        split_dir = os.path.join(path, 'splits', config['data.split'])
        split_file = lambda split: os.path.join(split_dir, f"{split}.txt")

        x_train, y_train = load_from_split(config['data.dataset'], config['data.version'], data_dir, split_file('train'))
        x_test, y_test = load_from_split(config['data.dataset'], config['data.version'], data_dir, split_file('test'))
        x_val, y_val = load_from_split(config['data.dataset'], config['data.version'], data_dir, split_file('val'))
    else:
        good_min = 40
        good_classes = []
        n_unique = len(np.unique(data[1]['y']))
        for i in range(n_unique):
            images = data[0][np.equal(i, data[1]['y'])]
            if len(images) >= good_min:
                good_classes = good_classes + [i]
                
        x = data[0][np.in1d(data[1]['y'], good_classes)]
        y = data[1]['y'][np.in1d(data[1]['y'], good_classes)]
        y_dict = dict(zip(np.unique(y), range(len(np.unique(y)))))
        y = np.vectorize(y_dict.get)(y)

        split = train_test_split if n_train_per_class <= 0 else train_test_split_balanced

        if n_train_per_class <= 0:
            x_train, x_test, y_train, y_test = split(x, y, train_size=train_size, test_size=test_size, stratify=y)
            x_train, x_val, y_train, y_val = split(x_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train)
        else:
            n_train_per_class = np.round(n_train_per_class * 1.6)
            x_train, x_test, y_train, y_test = split(np.array(x), np.array(y), train_size=train_size, test_size=test_size,
                                                    n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)
            x_train, x_val, y_train, y_val = split(x_train, y_train, n_train_per_class=n_train_per_class, test_size=0.2)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)