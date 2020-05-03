"""
Logic for evaluation procedure of saved model.
"""

import numpy as np
import tensorflow as tf
from densenet import densenet_model
from src.datasets import load
from sklearn.metrics import classification_report, accuracy_score
from src.engines.steps import steps
from src.utils.weighted_loss import weightedLoss
from src.transfer_learning.model import create_model

def eval(config):
    # Files path
    model_file_path = f"{config['model.path']}"
    data_dir = f"data/"

    _, _, test, nb_classes, image_shape, class_weights = load(config, datagen_flow=True)

    (test_gen, test_len, _) = test

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    if config['data.weight_classes']:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_object = weightedLoss(loss_object, class_weights)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()
    model = create_model(
        model_name=config['model.name'],
        weights=config['model.weights'],
        nb_classes=nb_classes,
        image_shape=image_shape,
        optimizer=optimizer,
        loss_object=loss_object,
    )
    model.load_weights(model_file_path)
    # model.summary()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    _, test_step = steps(model, loss_object, optimizer, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy, engine=config['engine'])

    batches = 0
    for test_images, test_labels in test_gen:
        test_step(test_images, test_labels)
        batches += 1
        if batches >= test_len / config['data.batch_size']:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

    losses.append(test_loss.result())
    accuracies.append(test_accuracy.result()*100)

    print('Test Loss: {} Test Acc: {}'.format(test_loss.result(), test_accuracy.result()*100))
