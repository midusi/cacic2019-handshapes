import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D

def create_model(nb_classes=None, image_shape=None, optimizer=None, loss_object=None):
    base_model = VGG16(include_top=False, weights=None, input_shape=image_shape)

    global_average_layer = GlobalAveragePooling2D()
    hidden_dense_layer = Dense(1024, activation='relu')
    prediction_layer = Dense(nb_classes, activation='softmax')
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        hidden_dense_layer,
        prediction_layer
    ])

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=["accuracy"])

    return model
