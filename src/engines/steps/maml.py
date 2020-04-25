import tensorflow as tf

def steps(model, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, model_copy, **kwargs):

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as test_tape:
            # test_tape.watch(model.trainable_variables)

            with tf.GradientTape() as train_tape:
                predictions = model(tf.cast(images, tf.float32), training=True)
                loss = loss_object(labels, predictions)
            
            gradients = train_tape.gradient(loss, model.trainable_variables)
            model_copy.set_weights(model.get_weights())
            optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

            predictions = model_copy(tf.cast(images, tf.float32), training=False)
            t_loss = loss_object(labels, predictions)

        gradients = test_tape.gradient(t_loss, model_copy.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(tf.cast(images, tf.float32), training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    return train_step, test_step
