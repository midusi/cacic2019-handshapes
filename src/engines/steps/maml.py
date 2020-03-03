import tensorflow as tf

def steps(model, loss_object, optimizer, train_loss, train_accuracy, val_loss, val_accuracy, lr_inner=0.01):
    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            # Step 5
            with tf.GradientTape() as tape:
                predictions = model(tf.cast(images, tf.float32), training=True)
                loss = loss_object(labels, predictions)
            # Step 6
            gradients = tape.gradient(loss, model.trainable_variables)
            k = 0
            model_copy = tf.keras.models.clone_model(model, images)
            for j in range(len(model_copy.layers)):
                model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                            tf.multiply(lr_inner, gradients[k]))
                model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                            tf.multiply(lr_inner, gradients[k+1]))
                k += 2
            # Step 8
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

    return train_step, test_step