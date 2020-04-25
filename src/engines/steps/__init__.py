from .steps import steps as normal_steps
from .maml import steps as maml_steps

def steps(*args, **kwargs):
    engine_steps = maml_steps if kwargs.get('engine') == 'maml' else normal_steps

    return engine_steps(*args, **kwargs)