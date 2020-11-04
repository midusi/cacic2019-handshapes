"""Load ciarp dataset"""

import os
import csv
import glob
import numpy as np
import handshape_datasets as hd
from os import listdir
from pathlib import Path


def read_csv(txt_path):
    with open(txt_path) as f:
        reader = csv.reader(f, delimiter=' ')
        filename, y = zip(*reader)
        y = np.array(list(map(int, y)))
    return filename, y


def load_folder(folder, txt_path):
    filenames, y = read_csv(txt_path)
    x = []
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        x.append(filepath)
    return x, y


def load_ciarp(data_dir, dataset, version='WithGabor'):
    """
    Load ciarp dataset.

    Returns (x, y): as dataset x and y.    

    """

    path = data_dir if data_dir else '/tf/data/{}/data'.format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    hd.load(dataset, Path(path))

    dataset_folder = os.path.join(path, 'ciarp/ciarp')
    version_string = version if version else 'WithGabor'
    folders = [f for f in os.scandir(
        dataset_folder) if f.is_dir() and f.path.endswith(version_string)]

    result = {}
    cant_images = 0
    for i, folder in enumerate(folders):  # Counts the amount of images
        images = list(
            filter(lambda x: ".db" not in x,
                   listdir(os.path.join(str(dataset_folder), folder.name)))
        )
        cant_images = len(images) + cant_images
    j = 0
    h = 0
    i = 0
    xtot = []
    ytot = np.zeros(cant_images, dtype='uint8')
    # Loop x to copy data into xtot
    for folder in folders:
        txt_name = f"{folder.name}.txt"
        txt_path = os.path.join(dataset_folder, txt_name)
        x, y = load_folder(folder, txt_path)
        for valuesy in y:
            ytot[j] = valuesy
            j += 1
        for valuesx in x:
            xtot.append(valuesx)
            h += 1
        i += 1
        result[folder.name] = (x, y)

    return xtot, ytot
