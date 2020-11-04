"""
Logic for evaluation procedure of saved model.
"""

import numpy as np
import tensorflow as tf
from densenet import densenet_model
from src.datasets import load
from sklearn.metrics import classification_report, accuracy_score
from tf_tools.engines.steps import steps
from tf_tools.weighted_loss import weighted_loss


def eval(config):
    # Files path
    model_file_path = f"{config['model.path']}"

    data = load(config, datagen_flow=True)

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    if config['data.weight_classes']:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_object = weighted_loss(loss_object, data["class_weights"])
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()
    model = densenet_model(classes=data["nb_classes"], shape=data["image_shape"],
                           growth_rate=config['model.growth_rate'], nb_layers=config['model.nb_layers'], reduction=config['model.reduction'])
    model.load_weights(model_file_path)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    _, _, test_step = steps(model, loss_object, optimizer, train_loss=train_loss, train_accuracy=train_accuracy,
                            test_loss=test_loss, test_accuracy=test_accuracy, engine=config['engine'])

    with tf.device(device_name):
        batches = 0
        for test_images, test_labels in data["test_gen"]:
            test_step(test_images, test_labels)
            batches += 1
            if batches >= data["test_size"] / config['data.batch_size']:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

    print('Test Loss: {} Test Acc: {}'.format(
        test_loss.result(), test_accuracy.result()*100))
