import os
import numpy as np
import tensorflow as tf


def train(model=None, epochs=10, batch_len=32, format_paths=True,
          train_gen=None, train_len=None, val_gen=None, val_len=None,
          train_loss=None, train_accuracy=None, test_loss=None, test_accuracy=None,
          val_loss=None, val_accuracy=None, train_step=None, test_step=None,
          checkpoint_path=None, max_patience=25, nb_classes=None,
          train_summary_writer=None, val_summary_writer=None, csv_output_file=None,
          optimizer=None, meta_optimizer=None, loss_object=None, lr=0.001,
          task_train_len=1, meta_train_len=1, **kwargs):

    min_loss = 100
    min_loss_acc = 0
    patience = 0

    results = 'epoch,loss,accuracy,val_loss,val_accuracy\n'

    if not meta_optimizer:
        meta_optimizer = tf.keras.optimizers.Adam()

    for epoch in range(epochs):
        while not ((batches + task_train_len + meta_train_len) >= train_len / batch_len):
            # get the weights of the initial model that will do the meta learning
            meta_model_weights = model.get_weights()

            # train on the task (one epoch)
            task_batches = 0
            for images, labels in train_gen:
                batches += 1
                train_step(images, labels)
                task_batches += 1
                if task_batches >= task_train_len:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break

            # test on the validation set the improvement achieved on one task for the meta learning
            meta_batches = 0
            sum_gradients = np.zeros_like(model.trainable_variables)
            for images, labels in train_gen:
                batches += 1
                gradients = meta_step(images, labels)
                gradients = np.array([np.array(x) for x in gradients])
                sum_gradients = sum_gradients + gradients
                meta_batches += 1
                if meta_batches >= meta_train_len:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break

            # set weights of the model to the weights of the original model
            model.set_weights(meta_model_weights)

            # update the weights of the meta learning model using the loss obtained from testing
            meta_optimizer.apply_gradients(
                zip(sum_gradients, model.trainable_variables))

        # get the weights of the initial model that will do the meta learning
        meta_model_weights = model.get_weights()

        # train on the task (one epoch)
        task_batches = 0
        for images, labels in train_gen:
            batches += 1
            train_step(images, labels)
            task_batches += 1
            if task_batches >= task_train_len:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # test the newly trained model on the training set
        batches = 0
        all_predictions = np.array([]).reshape(0, n_classes)
        all_labels = np.array([]).reshape(0, n_classes)
        for test_images, test_labels in test_gen:
            test_predictions = test_step(test_images, test_labels)
            all_predictions = np.vstack((all_predictions, test_predictions))
            all_labels = np.vstack(
                (all_labels, tf.one_hot(test_labels, n_classes)))
            batches += 1
            if batches >= test_len / batch_len:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # set weights of the model to the weights of the original model
        model.set_weights(meta_model_weights)

        results += '{},{},{},{},{}\n'.format(
            epoch,
            train_loss.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100)
        print('Epoch: {}, Train Loss: {}, Train Acc:{}, Test Loss: {}, Test Acc: {}'.format(
            epoch,
            train_loss.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100))

        if (val_loss.result() < min_loss):
            min_loss = val_loss.result()
            min_loss_acc = val_accuracy.result()
            patience = 0
            # serialize weights to HDF5
            if format_paths:
                checkpoint_path = checkpoint_path.format(
                    epoch=epoch, val_loss=min_loss, val_accuracy=min_loss_acc)
            model.save_weights(checkpoint_path)
        else:
            patience += 1

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            train_loss.reset_states()
            train_accuracy.reset_states()

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
            val_loss.reset_states()
            val_accuracy.reset_states()

        if patience >= max_patience:
            break

    file = open(csv_output_file, 'w')
    file.write(results)
    file.close()

    return min_loss, min_loss_acc
