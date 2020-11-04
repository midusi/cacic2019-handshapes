"""Load cifar100 dataset"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar100

def load_cifar100(dataset_name):
    """
    Load cifar100 dataset.

    Returns (x, y): as dataset x and y.

    """

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
