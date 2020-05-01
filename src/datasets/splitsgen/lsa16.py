"""Load lsa16 dataset"""

import os
import glob
import handshape_datasets as hd
from pathlib import Path

def load_lsa16(args):
    """
    Load lsa16 dataset.

    Returns (x, y): as dataset x and y.

    """

    path = '/tf/data/{}/data'.format(args['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)

    hd.load(args['dataset'], Path(path))

    dataDir = os.path.join(path, 'lsa16/lsa16_images/lsa32x32_nr_rgb_black_background', '*.{}'.format(args['ext']))
    images = map(os.path.basename, glob.glob(dataDir))

    x, y = [], []

    for filename in images:
        x.append(filename)
        y.append(int(filename.split("_")[0]))
    
    return x, y
