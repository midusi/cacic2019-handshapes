"""Load cifar10 dataset"""

import numpy as np
from pathlib import Path
from tensorflow.keras.datasets import cifar10

def load_cifar10(dataset_name):
    """
    Load cifar10 dataset.

    Returns (x, y): as dataset x and y.

    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)