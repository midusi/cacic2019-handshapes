from src.engines.steps import steps

from .train import train as normal_train
from .maml import train as maml_train

def train(*args, **kwargs):
    train_step, test_step = steps(*args, **kwargs)
    
    train_step=kwargs.get('train_step', train_step)
    test_step=kwargs.get('test_step', test_step)

    train_engine = maml_train if kwargs.get('engine') == 'maml' else normal_train

    return train_engine(train_step=train_step, test_step=test_step, **kwargs)
