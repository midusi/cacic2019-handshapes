"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import time
import tensorflow as tf
import tf_tools.engines.train as train_engine
from densenet import densenet_model
from src.datasets import load
from tf_tools.weighted_loss import weighted_loss


def train(config):
    data = load(config, datagen_flow=True, with_datasets=config['engine']=='maml')

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

    time_start = time.time()
    # Compiles a model, prints the model summary, and saves the model diagram into a png file.
    model = densenet_model(classes=data["nb_classes"], shape=data["image_shape"], growth_rate=config['model.growth_rate'],
                           nb_layers=config['model.nb_layers'], reduction=config['model.reduction'])
    # model.summary()

    # tf.keras.utils.plot_model(model, "{}/model.png".format(results_dir), show_shapes=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='val_accuracy')

    with tf.device(device_name):
        train_engine.train(
            log_info=config, model=model, batch_size=config['data.batch_size'],
            epochs=config['train.epochs'], max_patience=config['train.patience'],
            engine=config['engine'], lr=config['train.lr'],
            train_loss=train_loss, train_accuracy=train_accuracy,
            test_loss=val_loss, test_accuracy=val_accuracy,
            val_loss=val_loss, val_accuracy=val_accuracy,
            optimizer=optimizer, loss_object=loss_object,
            **data,
        )

    time_end = time.time()

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed % 3600//60
    sec = elapsed-min*60

    print(f"Training took: {h:.2f}h {min:.2f}m {sec:.2f}s!")
