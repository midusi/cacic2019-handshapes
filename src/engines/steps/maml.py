import tensorflow as tf

def steps(model, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, copied_model, **kwargs):

    def train_step(images, labels):
        with tf.GradientTape() as outer_tape:
            # outer_tape.watch(model.trainable_variables)
            
            with tf.GradientTape() as inner_tape:
                predictions = model(tf.cast(images, tf.float32), training=True)
                inner_loss = loss_object(labels, predictions)
            
            copied_model.set_weights(model.get_weights())
            gradients = inner_tape.gradient(inner_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, copied_model.trainable_variables))

            train_loss(inner_loss)
            train_accuracy(labels, predictions)

        predictions = copied_model(tf.cast(images, tf.float32), training=False)
        outer_loss = loss_object(labels, predictions)

        gradients = outer_tape.gradient(outer_loss, copied_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        test_loss(outer_loss)
        test_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(tf.cast(images, tf.float32), training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    return train_step, test_step