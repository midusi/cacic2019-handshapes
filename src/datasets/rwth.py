"""Load rwth dataset"""

import os
import numpy as np
import handshape_datasets as hd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.model_selection import train_test_split_balanced

def load_rwth(dataset_name, path=None, train_size=None, test_size=None, n_train_per_class=0, n_test_per_class=0):
    """
    Load rwth dataset.

    Returns (x, y): as dataset x and y.

    """

    if path == None:
        path = '/tf/data/{}/data'.format(dataset_name)

    if not os.path.exists(path):
        os.makedirs(path)
    data = hd.load(dataset_name, Path(path))
    
    good_min = 20
    good_classes = []
    n_unique = len(np.unique(data[1]['y']))
    for i in range(n_unique):
        images = data[0][np.equal(i,data[1]['y'])]
        if len(images) >= good_min:
            good_classes = good_classes + [i]
            
    x = data[0][np.in1d(data[1]['y'], good_classes)]
    y = data[1]['y'][np.in1d(data[1]['y'], good_classes)]
    y_dict = dict(zip(np.unique(y), range(len(np.unique(y)))))
    y = np.vectorize(y_dict.get)(y)

    split = train_test_split if n_train_per_class <= 0 else train_test_split_balanced

    if n_train_per_class <= 0:
        x_train, x_test, y_train, y_test = split(x, y, train_size=train_size, test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = split(x, y, n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class)

    return (x_train, y_train), (x_test, y_test)