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

def train_densenet(dataset_name = "rwth", rotation_range = 10, width_shift_range = 0.10,
          height_shift_range = 0.10, horizontal_flip = True, growth_rate = 128,
          nb_layers = [6,12], reduction = 0.0, lr = 0.001, epochs = 400,
          max_patience = 25, batch_size= 16, checkpoints = False, weight_classes = False,
          train_size=None, test_size=None):

    # log
    log_freq = 1
    save_freq = 40
    models_directory = 'models/'
    results_directory = 'results/'
    config_directory = 'config/'

    general_directory = "./results/"
    save_directory = general_directory + "{}/dense-net/".format(dataset_name)
    results = 'epoch,loss,accuracy,test_loss,test_accuracy\n'

    date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
    identifier = "{}-growth-{}-densenet-{}".format(
        '-'.join([str(i) for i in nb_layers]),
        growth_rate, 
        dataset_name
    ) + date

    summary_file = general_directory + 'summary.csv'

    # create summary file if not exists
    if not os.path.exists(summary_file):
        file = open(summary_file, 'w')
        file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
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

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    # create summary writers
    train_summary_writer = tf.summary.create_file_writer(save_directory + 'summaries/train/' + identifier)
    test_summary_writer = tf.summary.create_file_writer(save_directory +  'summaries/test/' + identifier)

    print("Starting training")

    min_loss = 100
    min_loss_acc = 0
    patience = 0

    for epoch in range(epochs):
        batches = 0
        for images, labels in train_gen:
            train_step(images, labels)
            batches += 1
            if batches >= train_len / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        batches = 0
        for test_images, test_labels in val_gen:
            test_step(test_images, test_labels)
            batches += 1
            if batches >= val_len / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        if (epoch % log_freq == 0):
            results += '{},{},{},{},{}\n'.format(epoch,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100)
            print ('Epoch: {}, Train Loss: {}, Train Acc:{}, Test Loss: {}, Test Acc: {}'.format(epoch,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))

            if (test_loss.result() < min_loss):    
                if not os.path.exists(save_directory + models_directory):
                    os.makedirs(save_directory + models_directory)
                # serialize weights to HDF5
                model.save_weights(save_directory + models_directory + "best{}.h5".format(identifier))
                min_loss = test_loss.result()
                min_loss_acc = test_accuracy.result()
                patience = 0
            else:
                patience += 1

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
                train_loss.reset_states()           
                train_accuracy.reset_states()           

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
                test_loss.reset_states()           
                test_accuracy.reset_states()   
                
        if checkpoints and epoch % save_freq == 0:
            if not os.path.exists(save_directory + models_directory):
                os.makedirs(save_directory + models_directory)
            # serialize weights to HDF5
            model.save_weights(save_directory + models_directory+"{}_epoch{}.h5".format(identifier,epoch))
            
        if patience >= max_patience:
            break

    if not os.path.exists(save_directory + results_directory):
        os.makedirs(save_directory + results_directory)

    file = open(save_directory + results_directory + 'results-'+ identifier + '.csv','w') 
    file.write(results) 
    file.close()

    if not os.path.exists(save_directory + config_directory):
        os.makedirs(save_directory + config_directory)

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

    file = open(save_directory + config_directory + identifier + '.json', 'w')
    file.write(json.dumps(config, indent=2))
    file.close()

    file = open(summary_file, 'a+')
    summary = "{}, {}, dense-net, {}, {}, {}\n".format(date,
                                                       dataset_name,
                                                       save_directory + config_directory + identifier + '.json',
                                                       min_loss,
                                                       min_loss_acc)
    file.write(summary)
    file.close()
