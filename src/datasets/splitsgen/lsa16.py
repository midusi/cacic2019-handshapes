"""Load lsa16 dataset"""

import os
import glob
import handshape_datasets as hd
from pathlib import Path

def load_lsa16(data_dir, dataset, version):
    """
    Load lsa16 dataset.

    Returns (x, y): as dataset x and y.    

    """

    path = data_dir if data_dir else '/tf/data/{}/data'.format(dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    hd.load(dataset, Path(path))

    version = 'lsa_nr_rgb' if version == 'colorbg' else 'lsa32x32_nr_rgb_black_background'

    dataDir = os.path.join(path, 'lsa16/lsa16_images/{}'.format(version), '*.png')

    x, y = [], []

    for filepath in glob.glob(dataDir):
        x.append(filepath)
        filename = os.path.basename(filepath)
        y.append(int(filename.split("_")[0]) - 1)
    
    return x, y
