import os
import numpy as np
from skimage import io, transform

def load_from_split(dataset, version, data_dir, split_file):
    # The data, split between train and test sets:
    if dataset == "Ciarp":
        x, y = load_ciarp_from_split(version, data_dir, split_file)
    elif dataset == "lsa16":
        x, y = load_lsa16_from_split(version, data_dir, split_file)
    elif dataset == "rwth":
        x, y = load_rwth_from_split(version, data_dir, split_file)
    else:
        raise ValueError("Unknow dataset: {}".format(dataset))

def load_ciarp_from_split(version, data_dir, split_file):
    n = sum(1 for _ in open(split_file))

    x = np.zeros((n, 38, 38, 1), dtype='uint8')
    y = np.zeros(n, dtype='uint8')

    with open(split_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            filepath, label = line.rstrip('\n').split(' ')
            image = io.imread(os.path.join(data_dir, filepath))
            image = image[:, :, np.newaxis]
            x[i, :] = image
            y[i] = int(label)

    return x, y

def load_lsa16_from_split(version, data_dir, split_file):
    n = sum(1 for _ in open(split_file))

    x = np.zeros((n, 32, 32, 3), dtype='uint8')
    y = np.zeros(n, dtype='uint8')

    with open(split_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            filepath, label = line.rstrip('\n').split(' ')
            image = io.imread(os.path.join(data_dir, filepath))
            if version == "colorbg":
                image = transform.resize(image, (32, 32), preserve_range=True, mode="reflect",
                                         anti_aliasing=True)
            x[i, :, :, :] = image
            y[i] = int(label)

    return x, y

def load_rwth_from_split(version, data_dir, split_file):
    n = sum(1 for _ in open(split_file))

    y = np.zeros(n, dtype='uint8')
    x = []

    with open(split_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            filepath, label = line.rstrip('\n').split(' ')
            image = io.imread(os.path.join(data_dir, filepath))
            image = image[np.newaxis, :, :]
            if len(image.shape) == 2:
                pass
            x.append(im)
            y[i] = int(label)

    x = np.vstack(x)

    return x, y