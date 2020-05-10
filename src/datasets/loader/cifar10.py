"""Load cifar10 dataset"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

def load_cifar10(dataset_name):
    """
    Load cifar10 dataset.

    Returns (x, y): as dataset x and y.

    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
