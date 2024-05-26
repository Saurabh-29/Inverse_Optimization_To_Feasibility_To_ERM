
# imports
import torch
import numpy as np
from copy import deepcopy
import time
from functorch import make_functional, grad, make_functional_with_buffers

class Armijo(torch.optim.Optimizer):
    def __init__(self, model, param_dict):
        super().__init__(model.parameters(), {})
        self.func, self.params, self.buffer = make_functional_with_buffers(model)

        self.model = model

        # create some local tools
        self.m = param_dict["m"] if "m" in param_dict else 1
        self.lr = param_dict["lr"]
        self.b_fwd = param_dict["b_fwd"]
        self.b_bwd = param_dict["b_bwd"]
        self.c = param_dict["c"]
        self.reset_lr_on_step = param_dict["reset_lr_on_step"] if "reset_lr_on_step" in param_dict else False
    
        self.init_step_size = self.lr
        self.state['steps'] = 0
        self.prev_lr = self.lr

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
    
    def get_squared_grad_norm(self, parameters):
        """
        args:
            parameters: model parameters
        
        returns: the squared norm of the gradient.
        """
        total_norm = 0
        for p in parameters:
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm


    def reset_prev_lr(self):
        self.prev_lr = self.init_step_size
    
    

    def update_params(self, lr, params):
        """
        args:
            lr: learning rate
        Updates the model parameters by -lr*gradient
        """
        with torch.no_grad():
            for param in params:
                if param.grad is not None:
                    param.data -= lr * param.grad.data

    def step(self, model, input, target, closure):

        with torch.no_grad():
            z = input
            grad_norm = self.get_squared_grad_norm(model.parameters())

            base_lr = self.prev_lr
            base_loss = closure(self.func(self.params, self.buffer, z), target)
            # print("base loss is ", base_loss.item())

            lr = self.init_step_size

            while True:
                for p1, p2 in zip(model.parameters(), self.params):
                    if p1.grad is not None:
                        p2.data = p1.data - lr*p1.grad
                
                # loss = self.criterion_with_target(target, func(params, z))
                current_loss = closure(self.func(self.params, self.buffer, z), target)
                c_armijo = self.c
                # print("current_loss", current_loss, "base_loss", base_loss, "grad_norm", grad_norm, "lr", lr)
                if current_loss <= base_loss - c_armijo*lr*grad_norm:
                    self.update_params(lr, model.parameters())
                    self.prev_lr = lr*self.b_fwd
                    # print("successs")
                    # exit()
                    return
                if lr < 1e-6:
                    break
                lr *= self.b_bwd

            