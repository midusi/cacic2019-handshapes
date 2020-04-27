import tensorflow as tf

from src.engines.steps import steps
from .train import train

def train(*args, **kwargs):
    model = kwargs.get('model')
    copied_model = tf.keras.models.clone_model(model) if kwargs.get('engine') == 'maml' else None

    train_step, test_step = steps(*args, copied_model=copied_model, **kwargs)
    
    kwargs['train_step'] = kwargs.get('train_step', train_step)
    kwargs['test_step'] = kwargs.get('test_step', test_step)

    return train(*args, **kwargs)
