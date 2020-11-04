"""
Logic for evaluation procedure of saved model.
"""

import numpy as np
import tensorflow as tf

# tf.config.gpu.set_per_process_memory_growth(True)
# tf.debugging.set_log_device_placement(True)

from protonet.models import Prototypical
from src.protonet.datasets import load


def eval(config):
    n_support = config['data.test_support']
    n_query = config['data.test_query']
    w, h, c, = list(map(int, config['model.x_dim'].split(',')))
    model = Prototypical(n_support, n_query, w, h, c, nb_layers=config['model.nb_layers'], nb_filters=config['model.nb_filters'])
    model_path = f"{config['model.save_path'].format(config['model.type'])}"
    model.load(model_path)

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Load data from disk
    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ['test'])
    test_loader = ret['test']

    # Metrics to gather
    test_loss = tf.metrics.Mean(name='test_loss')
    test_acc = tf.metrics.Mean(name='test_accuracy')

    def calc_loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    with tf.device(device_name):
        for i_episode in range(config['data.episodes']):
            support, query = test_loader.get_next_episode()
            # if (i_episode+1)%50 == 0: 
            #    print("Episode: ", i_episode + 1)
            loss, acc = calc_loss(support, query)
            test_loss(loss)
            test_acc(acc)

    print('Test Loss: {} Test Acc: {}'.format(test_loss.result(), test_acc.result()*100))

if __name__ == "__main__":
    pass
