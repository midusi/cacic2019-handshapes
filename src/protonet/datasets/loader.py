import os
import glob
import numpy as np
import tensorflow as tf
import handshape_datasets as hd
from pathlib import Path
from src.datasets import load as load_dataset
from tf_tools.model_selection import train_test_split_balanced
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query, x_dim):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.x_dim = x_dim

    def get_next_episode(self):
        w, h, c = self.x_dim
        support = np.zeros(
            [self.n_way, self.n_support, w, h, c], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, w, h, c], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            n_samples = self.data[i_class].shape[0]
            selected = np.random.permutation(
                n_samples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class][selected[:self.n_support]]
            query[i] = self.data[i_class][selected[self.n_support:]]

        return support, query


def load(data_dir, config, splits):
    """
    Load specific dataset.

    Args:
        data_dir (str): path to the dataset directory.
        config (dict): general dict with settings.
        splits (list): list of strings 'train'|'val'|'test'.

    Returns (dict): dictionary with keys 'train'|'val'|'test'| and values
    as tensorflow Dataset objects.

    """
    dataset_path = '/tf/data/{}'.format(config['data.dataset'])

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    data = load_dataset(config, with_datasets=True)

    ret = {}

    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test']:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']

        # n_support (number of support samples per class)
        if split in ['val', 'test']:
            n_support = config['data.test_support']
        else:
            n_support = config['data.train_support']

        # n_query (number of query samples per class)
        if split in ['val', 'test']:
            n_query = config['data.test_query']
        else:
            n_query = config['data.train_query']

        x, y = data[f"x_{split}"], data[f"y_{split}"]

        # _, amountPerClass = np.unique(y, return_counts=True)

        i = np.argsort(y)
        y = y[i]
        x = x[i, :, :, :]

        if config['model.type'] in ['augmentation']:
            for index in i:
                x[index, :, :, :] = data[f"{split}_datagen"].apply_transform(
                    x[index], data[f"{split}_datagen_args"])

        data = [[] for i in range(data["nb_classes"])]
        for index in i:
            data[y[index]].append(x[index])

        data_loader = DataLoader(np.array([np.array(images) for images in data]),
                                 n_classes=data["nb_classes"],
                                 n_way=n_way,
                                 n_support=n_support,
                                 n_query=n_query,
                                 x_dim=data["image_shape"])

        ret[split] = data_loader

    return ret
