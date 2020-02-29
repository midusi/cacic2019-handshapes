"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.datasets import load
from src.vgg.model import create_model
from src.utils.weighted_loss import weightedLoss

def train(config):
    np.random.seed(2020)
    tf.random.set_seed(2020)

    # Useful data
    now = datetime.now()
    now_as_str = now.strftime('%y_%m_%d-%H:%M:%S')

    # Output files
    checkpoint_path = f"{config['model.save_path']}"
    config_path = f"{config['output.config_path'].format(now_as_str)}"
    csv_output_path = f"{config['output.train_path'].format(now_as_str)}"
    tensorboard_summary_dir = f"{config['summary.save_path']}"
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

    file = open(f"{csv_output_path}", 'w') 
    file.write("")
    file.close()

    # Create folder for config
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # generate config file
    file = open(config_path, 'w')
    file.write(json.dumps(config, indent=2))
    file.close()

    # create summary file if not exists
    if not os.path.exists(summary_path):
        file = open(summary_path, 'w')
        file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
        file.close()

    # Data loader
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train, val, _, nb_classes, image_shape, class_weights = load(
        dataset_name=config['data.dataset'],
        batch_size=config['data.batch_size'],
        train_size=config['data.train_size'],
        test_size=config['data.test_size'],
        weight_classes=config['data.weight_classes'],
        rotation_range=config['data.rotation_range'],
        width_shift_range=config['data.width_shift_range'],
        height_shift_range=config['data.height_shift_range'],
        horizontal_flip=config['data.horizontal_flip'],
        datagen_flow=True,
    )

    (train_gen, _, _) = train
    (val_gen, _, _) = val

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_summary_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='epoch'
    )

    logs_callback = tf.keras.callbacks.CSVLogger(
        csv_output_path,
        separator=',',
        append=False
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['train.patience']
    )

    if config['data.weight_classes']:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_object = weightedLoss(loss_object, class_weights)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    time_start = time.time()
    # Compiles a model, prints the model summary, and saves the model diagram into a png file.
    model = create_model(nb_classes=nb_classes, image_shape=image_shape, optimizer=optimizer, loss_object=loss_object)
    
    model.summary()
    tf.keras.utils.plot_model(model, "vgg_model.png", show_shapes=True)

    # Trains the model.
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['train.epochs'],
        use_multiprocessing=True,
        callbacks=[tensorboard_callback, logs_callback, model_checkpoint_callback, early_stop]
    )

    time_end = time.time()

    # Evaluates on test data.
    loss, acc = model.evaluate(val_gen)
    print("Evaluation finished!")

    summary = "{}, {}, df_model, {}, {}, {}\n".format(now_as_str, config['data.dataset'], config_path, loss, acc)
    print(summary)

    file = open(summary_path, 'a+') 
    file.write(summary)
    file.close()

    # Runs prediction on test data.
    predictions = model.predict(val_gen)
    print("Predictions on test data:")
    print(predictions)

    model_path = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=checkpoint_path)

    if not model_path:
        print("Skipping evaluation. No checkpoint found in: {}".format(checkpoint_dir))
    else:
        model_from_saved = tf.keras.models.load_model(model_path)
        model_from_saved.summary()

        # Runs test data through the reloaded model to make sure the results are same.
        predictions_from_saved = model_from_saved.predict(val_gen)

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed%3600//60
    sec = elapsed-min*60

    print(f"Training took: {h:.2f}h {min:.2f}m {sec:.2f}s!")

