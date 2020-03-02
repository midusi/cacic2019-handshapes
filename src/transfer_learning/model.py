import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3, VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201

models = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'InceptionV3': InceptionV3,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
}

def create_model(model_name=None, nb_classes=None, image_shape=None, optimizer=None, loss_object=None, weights=None, bm_trainable=False):
    if model_name not in models:
        print('Error: Wrong model name. VGG16 will be used as default.')
        model_name = 'VGG16'
    
    weights = None if weights == '' else weights

    base_model =  models[model_name](include_top=False, weights=weights, input_shape=image_shape)
    base_model.trainable = bm_trainable
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
    model.compile(optimizer=optimizer, loss=loss_object, metrics=["accuracy", "sparse_categorical_accuracy"])

    return model
