from densenet import densenet_model

def create_base_model(model_name=None, image_shape=None):
    if model_name == 'DenseNet':
        base_model = densenet_model(shape=image_shape, growth_rate=64, nb_layers=[6], reduction=0.5, with_output_block=False)
    else:
        print('Error: Wrong model name. DenseNet will be used as default.')
        base_model = densenet_model(shape=image_shape, growth_rate=64, nb_layers=[6], reduction=0.5, with_output_block=False)

    return base_model
