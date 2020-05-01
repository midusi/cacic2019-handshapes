"""Load ciarp dataset"""

import os
import glob
import handshape_datasets as hd
from pathlib import Path

def read_csv(txt_path):
    with open(txt_path) as f:
        reader = csv.reader(f, delimiter=' ')
        filename, y = zip(*reader)
        y = np.array(list(map(int, y)))
    return filename, 

def load_folder(folder, txt_path):
    filenames, y = read_csv(txt_path)
    x = []
    for filename in filenames:
        filepath = os.path.join(folder.path, filename)
        x.append(filepath)
    return x, y

def extract_ciarp_classes(args):
    """
    Load ciarp dataset.

    Returns (x, y): as dataset x and y.

    """

    path = '/tf/data/{}/data'.format(args['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)

    hd.load(args['dataset'], Path(path))

    dataset_folder = os.path.join(path , 'ciarp')
    folder_image=path
    version_string='WithGabor'
    folders = [f for f in os.scandir(dataset_folder) if f.is_dir() and f.path.endswith(version_string)]

    result={}
    cant_images=0
    for i, folder in enumerate(folders):  #Counts the amount of images
        images = list(
            filter(lambda x: ".db" not in x,
                    listdir(os.path.join(str(dataset_folder), folder.name)))
        )
        cant_images = len(images) + cant_images
    j = 0
    h = 0
    i = 0
    subject = np.zeros(cant_images)
    xtot=np.zeros((cant_images, 38, 38, 1), dtype="uint8")
    ytot=np.zeros(cant_images,dtype='uint8')
    #Loop x to copy data into xtot
    for folder in folders:
        txt_name=f"{folder.name}.txt"
        txt_path=os.path.join(dataset_folder,txt_name)
        x,y=load_folder(folder,txt_path)
        for valuesy in y:
            ytot[j] = valuesy
            j += 1
        for valuesx in x:
            xtot[h] = valuesx
            subject[h] = i
            h += 1
        i += 1
        result[folder.name] = (x,y)
    
    return xtot, ytot
