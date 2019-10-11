import numpy as np

class Evaluator(object):

    def __init__(self, height, width, f_outputs):
        self.loss_value = None
        self.grads_values = None
        self.height = height
        self.width = width
        self.f_outputs = f_outputs

    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.height, self.width, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values


    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values