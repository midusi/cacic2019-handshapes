import os
import sys
import json
import getopt
import numpy as np
import tensorflow as tf
import src.engines.train as train_engine
from datetime import datetime
from densenet import densenet_model
from src.datasets import load
from src.utils.weighted_loss import weightedLoss

def train_densenet(dataset_name="rwth", rotation_range=10, width_shift_range=0.10,
                   height_shift_range=0.10, horizontal_flip=True, growth_rate=128,
                   nb_layers=[6,12], reduction=0.0, lr=0.001, epochs=400,
                   max_patience=25, batch_size=16, checkpoints=False, weight_classes=False,
                   train_size=None, test_size=None):

    # Useful data
    now = datetime.now()
    now_as_str = now.strftime('%y_%m_%d-%H:%M:%S')

    identifier = "{}-growth-{}-densenet-{}".format(
        '-'.join([str(i) for i in nb_layers]),
        growth_rate, 
        dataset_name
    ) + now_as_str

    results_root = '/develop/results'
    save_directory = results_root + "/{}/dense_net/".format(dataset_name)
    summary_file_path = results_root + '/summary.csv'

    # Output files
    checkpoint_path = save_directory + "/checkpoints/checkpoint.{}.h5".format(identifier)
    config_path = save_directory + "/config/" + identifier + '.json'
    csv_output_path = save_directory + '/results-'+ identifier + '.csv'
    train_summary_file_path = save_directory + '/summaries/train/' + identifier
    test_summary_file_path = save_directory + '/summaries/test/' + identifier
    summary_path = f"results/summary.csv"
    
    # Output dirs
    data_dir = f"data/"
    checkpoint_dir = checkpoint_path[:checkpoint_path.rfind('/')]
    config_dir = config_path[:config_path.rfind('/')]
    results_dir = csv_output_path[:csv_output_path.rfind('/')]

    # Create folder for model
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create output for train process
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # create summary file if not exists
    if not os.path.exists(summary_file_path):
        file = open(summary_file_path, 'w')
        file.write("datetime, model, config, loss, accuracy\n")
        file.close()

    print("Hyperparameters set")

    train, val, _, nb_classes, image_shape, class_weights = load(
        dataset_name, batch_size=batch_size, datagen_flow=True,
        train_size=train_size, test_size=test_size,
        weight_classes=weight_classes,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
    )

    (train_gen, train_len, _) = train
    (val_gen, val_len, _) = val

    model = densenet_model(classes=nb_classes, shape=image_shape, growth_rate=growth_rate, nb_layers=nb_layers, reduction=reduction)

    print("DenseNet Model created")

    if weight_classes:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_object = weightedLoss(loss_object, class_weights)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(tf.cast(images, tf.float32), training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(tf.cast(images, tf.float32), training=False)
        t_loss = loss_object(labels, predictions)
        val_loss(t_loss)
        val_accuracy(labels, predictions)

    # create summary writers
    train_summary_writer = tf.summary.create_file_writer(train_summary_file_path)
    val_summary_writer = tf.summary.create_file_writer(test_summary_file_path)

    loss, acc = train_engine.train(
        model=model, batch_size=batch_size,
        epochs=epochs, max_patience=max_patience,
        train_gen=train_gen, train_len=train_len, val_gen=val_gen, val_len=val_len,
        train_loss=train_loss, train_accuracy=train_accuracy,
        val_loss=val_loss, val_accuracy=val_accuracy,
        train_step=train_step, test_step=test_step,
        checkpoint_path=checkpoint_path,
        train_summary_writer=train_summary_writer,
        val_summary_writer=val_summary_writer,
        csv_output_file=csv_output_path,
        format_paths=False,
    )

    config = {
        'data.dataset_name': dataset_name, 
        'data.rotation_range': rotation_range, 
        'data.width_shift_range': width_shift_range, 
        'data.height_shift_range': height_shift_range, 
        'data.horizontal_flip': horizontal_flip, 
        'model.growth_rate': growth_rate, 
        'model.nb_layers': nb_layers, 
        'model.reduction': reduction, 
        'train.lr': lr, 
        'train.epochs': epochs, 
        'train.max_patience': max_patience, 
        'train.batch_size': batch_size, 
    }

    file = open(config_path, 'w')
    file.write(json.dumps(config, indent=2))
    file.close()

    file = open(summary_file_path, 'a+')
    summary = "{}, {}, dense_net, {}, {}, {}\n".format(now_as_str,
                                                       dataset_name,
                                                       config_path,
                                                       loss,
                                                       acc)
    file.write(summary)
    file.close()
