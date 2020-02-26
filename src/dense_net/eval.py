import os
import sys
import json
import getopt
import numpy as np
import tensorflow as tf
from datetime import datetime
from densenet import densenet_model
from src.datasets import load
from src.utils.weighted_loss import weightedLoss

def eval_densenet(dataset_name = "rwth", growth_rate = 128, nb_layers = [6,12],
                  reduction = 0.0, max_patience = 25, batch_size = 16, checkpoints = False,
                  weight_classes = False, model_path = "", test_size=0.25, train_size=0.75):

    np.random.seed(2019)
    tf.random.set_seed(2019)

    # log
    log_freq = 1

    print("Hyperparameters set")

    _, _, test, nb_classes, image_shape, class_weights = load(
        dataset_name, batch_size=batch_size, datagen_flow=True,
        train_size=train_size, test_size=test_size,
        weight_classes=weight_classes,
    )

    (test_gen, test_features, _) = test

    model = densenet_model(classes=nb_classes, shape=image_shape, growth_rate=growth_rate, nb_layers=nb_layers, reduction=reduction)
    model.load_weights(model_path)

    print("DenseNet Model created")

    if weight_classes:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_object = weightedLoss(loss_object, class_weights)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        predictions = model(tf.cast(images, tf.float32), training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    print("Starting evaluation")

    batches = 0
    for test_images, test_labels in test_gen:
        test_step(test_images, test_labels)
        batches += 1
        if batches >= test_features / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

    print ('Test Loss: {} Test Acc: {}'.format(test_loss.result(),
                                               test_accuracy.result()*100))
