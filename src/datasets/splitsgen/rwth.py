"""Load rwth dataset"""

import os
import glob
import platform
import numpy as np
import handshape_datasets as hd
from pathlib import Path


def load_rwth(data_dir, dataset, version):
    """
    Load rwth dataset.

    Returns (x, y): as dataset x and y.    

    """

    path = data_dir if data_dir else '/tf/data/{}/data'.format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    hd.load(dataset, Path(path))

    extracted_folderpath = os.path.join(
        path, 'rwth-phoenix', "ph2014-dev-set-handshape-annotations")
    metadata_path = os.path.join(
        extracted_folderpath, "3359-ph2014-MS-handshape-annotations.txt")
    search = "*"
    replace1 = "_"
    with open(metadata_path) as f:
        lines = f.readlines()
    lines = [x.strip().split(" ") for x in lines]
    for l in lines:
        assert (len(l) == 2)
    images_paths = [x[0] for x in lines]
    images_class_names = [x[1] for x in lines]
    if platform.system() == "Windows":
        i = 0
        for im in images_paths:
            images_paths[i] = im.replace(search, replace1)
            i = i+1
    classes = sorted(list(set(images_class_names)))
    y = np.array([classes.index(name) for name in images_class_names])
    paths = [os.path.join(extracted_folderpath, path) for path in images_paths]
    x = []
    for filepath in paths:
        x.append(filepath)

    x = np.array(x)

    good_min = 40
    good_classes = []
    n_unique = len(np.unique(y))
    for i in range(n_unique):
        images = x[np.equal(i, y)]
        if len(images) >= good_min:
            good_classes = good_classes + [i]

    x = x[np.in1d(y, good_classes)]
    y = y[np.in1d(y, good_classes)]
    y_dict = dict(zip(np.unique(y), range(len(np.unique(y)))))
    y = np.vectorize(y_dict.get)(y)

    return x, y
