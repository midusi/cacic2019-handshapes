from .steps import steps as normal_steps
from .maml import steps as maml_steps

def steps(model, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, engine=None, lr=0.001):
    if engine == 'maml':
        return normal_steps(model, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy)
    return maml_steps(model, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, lr_inner=lr)
