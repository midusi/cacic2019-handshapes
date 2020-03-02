"""
Logic for evaluation procedure of saved model.
"""

import tensorflow as tf
import tensorflowjs as tfjs
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, accuracy_score

from src.datasets import load

def eval(config):
    # Files path
    model_file = f"{config['model.path']}"
    data_dir = f"data/"

    _, _, test, nb_classes, image_shape, class_weights = load(
        dataset_name=config['data.dataset'],
        batch_size=config['data.batch_size'],
        train_size=config['data.train_size'],
        test_size=config['data.test_size'],
        weight_classes=config['data.weight_classes'],
        datagen_flow=True,
    )

    (test_gen, _, _) = test

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    model = tf.keras.models.load_model(model_file)
    model.summary()

    predictions = model.predict(test_gen)
