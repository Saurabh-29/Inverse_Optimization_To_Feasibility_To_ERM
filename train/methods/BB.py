import torch
import torch.nn as nn 
import pickle
import numpy as np
from scipy.optimize import linprog
import sys
sys.path.append("../")
# import Algorithm.LinearProgramMethod as lpm
from functorch import make_functional, grad

from utils import get_squared_grad_norm, get_loss_with_LP_variables, get_test_loss, get_sol_from_LP_variables
from data_utils import loadData
from models import get_model
from utils import *
import time 
from Solver import Solver


class BB():
    def __init__(self, config):
        """_summary_

        Args:
            config (_type_): _description_
            
        """
        super(BB, self).__init__()
        self.model_params = config["model_params"]
        self.model = get_model(config["model_params"])
        self.config = config
        self.data_path = config["data_path"]
        self.data = loadData(self.data_path, config)

        self.train_params = config["train_params"]

        self.train_data = self.data.train_data
        self.valid_data = self.data.valid_data
        self.test_data = self.data.test_data

        self.hp_lamb = config["method"]["hp_lamb"]
        self.solver = ShortestPath.apply
        self.optimizer_name, self.optimizer = return_optimizer_from_string(config["optimizer"], self.model)

        self.train_data, self.valid_data, self.test_data = kernelize_data(config, self.train_data, self.valid_data, self.test_data)

        lp_params = {"A": self.train_data["A"], "b": self.train_data["b"], "problem": "LP", "max_cap_on_pred": config["method"]["max_cap_on_pred"]}
        self.lp_solver = Solver(lp_params)


    def trainer(self):
        input = torch.from_numpy(self.train_data["z"]).float()
        target = torch.from_numpy(self.train_data["sol"]).float()
        prev_lr = self.config["optimizer"]["lr"]
        log_data = []
        a2 =0 
        for epoch in range(self.train_params["num_epochs"]):
            a1 = time.time()
            c_t = self.model(input)
            x_pred = self.solver(c_t, self.train_data, self.hp_lamb)
            loss = self.criterion(target, x_pred)
            self.set_zero_grad()
            loss.backward()
            
            if self.optimizer_name == "armijo":
                lr = self.get_lr_by_armijo(self.train_data, prev_lr)
                self.update_params(lr)
                prev_lr = 1.5*lr
            else:
                self.optimizer.step()
                lr = self.config["optimizer"]["lr"]
            
            log_data, train_loss = update_log_data(log_data, self.config, self.model, lr, loss, epoch,
                                self.train_data, self.valid_data, None)
                                
            a2 += time.time() -a1
            print("Epoch: {}, {}, time: {}".format(epoch, train_loss, a2))

        save_log_file(log_data, self.config, lr)
        save_model(self.model, self.optimizer, self.config, lr)

    def set_zero_grad(self):
        self.model.zero_grad()
    
    def update_params(self, lr):
        with torch.no_grad():
            for param in self.model.parameters():
                param.data -= lr * param.grad.data
    
    def criterion(self, x_target, x_pred):
        return (0.5*((x_target - x_pred)**2).sum()) / x_target.shape[0]
    
    def inference(self, input, data):
        with torch.no_grad():
            return self.solver(self.model(input), data)
    
    def get_lr_by_armijo(self, data, prev_lr):
        with torch.no_grad():
            sol, z = data["sol"], data["z"]
            z = torch.from_numpy(z).float()
            sol = torch.from_numpy(sol).float()

            func, params = make_functional(self.model)
                
            grad_norm = get_squared_grad_norm(self.model.parameters())
            base_lr = prev_lr
            base_loss = self.criterion(sol, self.solver(func(params, z), data, self.hp_lamb))

            lr = base_lr
            while True:
                for p1, p2 in zip(self.model.parameters(), params):
                    # p1.grad -= torch.mean(p1.grad)
                    p2.data = p1.data - lr*p1.grad
                
                loss = self.criterion(sol, self.solver(func(params, z), data, self.hp_lamb))
                if loss <= base_loss - 0.5*lr*grad_norm:
                    break
                if lr < 1e-10:
                    break
                lr *= 0.9
            return lr

    def criterion_with_c_pred(self, c_pred, x_target):
        x_pred = self.solver(c_pred, self.lp_solver, self.hp_lamb)
        return (0.5*((x_target - x_pred)**2).sum())/ x_target.shape[0]

    def get_loss(self):
        return self.criterion_with_c_pred
    
    def get_solver(self):
        return self.lp_solver
    
    def forward(self, x):
        c_t = self.model(x)
        return c_t
    
    def get_target_fn(self):
        return self.get_target
        
    
    def get_target(self, c_pred, sol):
        """ To ensure consistency with the other methods, we return the target as a tensor"""
        return torch.from_numpy(sol).float().to(c_pred.device)

class ShortestPath(torch.autograd.Function):
    """ Define a new function for shortest path to avoid the legacy autograd
    error """

    @staticmethod
    def forward(ctx, c_pred, solver, lambda_val):
        ctx.weights = c_pred.detach().cpu().numpy()
        ctx.solver = solver
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(c_pred)
        ctx.suggested_tours = get_sol_from_LP_variables(solver, ctx.weights)
        return torch.from_numpy(ctx.suggested_tours).float().to(c_pred.device)


    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        weights_cpu = weights.detach().cpu().numpy()

        grad_output_numpy = grad_output.detach().cpu().numpy()

        assert grad_output_numpy.shape == ctx.suggested_tours.shape

        weights_prime = weights_cpu + ctx.lambda_val * grad_output_numpy
        weights_prime = np.maximum(weights_prime, 0.0)
        
        better_paths = get_sol_from_LP_variables(ctx.solver, weights_prime)
        gradient = -(ctx.suggested_tours - better_paths) / ctx.lambda_val

        return torch.from_numpy(gradient).to(grad_output.device), None, None
