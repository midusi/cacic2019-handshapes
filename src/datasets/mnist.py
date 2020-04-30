"""Load mnist dataset"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.datasets import mnist

def load_mnist(dataset_name):
    """
    Load mnist dataset.

    Returns (x, y): as dataset x and y.

    """

    if path == None:
        path = '/tf/data/{}/data'.format(dataset_name)

    if not os.path.exists(path):
        os.makedirs(path)
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)
