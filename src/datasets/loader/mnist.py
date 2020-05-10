"""Load mnist dataset"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

def load_mnist(dataset_name):
    """
    Load mnist dataset.

    Returns (x, y): as dataset x and y.

    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
