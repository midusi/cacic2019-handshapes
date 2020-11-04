import tensorflow as tf
from densenet import densenet_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Concatenate, Conv2D
from tensorflow.keras.applications import InceptionV3, VGG16, VGG19, densenet, DenseNet201, DenseNet201

models = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'InceptionV3': InceptionV3,
    'densenet': densenet,
    'DenseNet201': DenseNet201,
    'DenseNet': None,
}


def create_model(model_name=None, nb_classes=None, image_shape=None, optimizer=None, loss_object=None, weights=None, bm_trainable=False):
    if model_name not in models:
        print('Error: Wrong model name. VGG16 will be used as default.')
        model_name = 'VGG16'

    weights = None if weights == '' else weights
    w, h, c = image_shape

    if model_name == 'DenseNet':
        base_model = densenet_model(shape=(w, h, 3), growth_rate=64, nb_layers=[
                                    6, 12], reduction=0.5, with_output_block=False)
        if weights != None:
            base_model.load_weights(weights, by_name=True)
    else:
        base_model = models[model_name](
            include_top=False, weights=weights, input_shape=(w, h, 3))
        base_model.trainable = bm_trainable

    global_average_layer = GlobalAveragePooling2D()
    hidden_dense_layer = Dense(1024, activation='relu')
    prediction_layer = Dense(nb_classes, activation='softmax')
    model = tf.keras.Sequential()

    if c < 3:
        model.add(Input(shape=(w, h, 1)))
        model.add(Conv2D(3, (3, 3), padding='same'))

    model.add(base_model)
    model.add(global_average_layer)
    model.add(hidden_dense_layer)
    model.add(prediction_layer)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer, loss=loss_object)

    return model
