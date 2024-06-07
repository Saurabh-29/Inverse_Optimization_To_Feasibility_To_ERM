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
from models import LinearModel, get_model
from utils import *
import time
from Solver import Solver

class QPTL():
    def __init__(self, config):
        super(QPTL, self).__init__()
        self.model_params = config["model_params"]
        # self.model = LinearModel(config["model_params"])
        self.model = get_model(config["model_params"])
        self.config = config
        self.data_path = config["data_path"]
        self.data = loadData(self.data_path, config)

        self.train_params = config["train_params"]

        self.train_data = self.data.train_data
        self.valid_data = self.data.valid_data
        self.test_data = self.data.test_data

        self.hp_lamb = self.config.method["hp_lamb"]
        self.block_matrix = self.get_block_matrix(self.train_data, self.hp_lamb)
        self.solver = ShortestPath.apply
        self.optimizer_name, self.optimizer = return_optimizer_from_string(config["optimizer"], self.model)

        self.train_data, self.valid_data, self.test_data = kernelize_data(config, self.train_data, self.valid_data, self.test_data)
        lp_params = {"A": self.train_data["A"], "b": self.train_data["b"], "problem": "LP", "gamma": self.hp_lamb}
        self.qp_solver = Solver(lp_params)

        if self.config.task == "portfolio":
            lp_params_2 = {"A": self.train_data["A"], "b": self.train_data["b"], "problem": "LP", "gamma": self.hp_lamb}
        else:
            lp_params_2 = {"A": self.train_data["A"], "b": self.train_data["b"], "problem": "LP", "gamma": 0.0}
        self.lp_solver = Solver(lp_params_2)

    def trainer(self):
        input = torch.from_numpy(self.train_data["z"]).float()
        target = torch.from_numpy(self.train_data["sol"]).float()
        prev_lr = self.config["optimizer"]["lr"]
        log_data = []
        a2 = 0
        for epoch in range(self.train_params["num_epochs"]):
            a1 = time.time()
            c_t = self.model(input)
            x_pred = self.solver(c_t, self.train_data, self.block_matrix, self.hp_lamb)
            loss = self.criterion(x_pred, target)
            print("Loss", loss.item())
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

    def get_block_matrix(self, data, lamb):
        n = data["A"].shape[1]
        m = data["A"].shape[0]
        current_lamb = np.zeros(n)
        current_x = np.zeros(n)
        I = np.eye(n)
        gamma_I = 2*lamb*np.eye(n)
        D_lamb = -np.diag(current_lamb)
        D_x = -np.diag(current_x)
        A = data["A"]
        top_row = np.hstack((gamma_I, D_lamb, A.T))
        middle_row = np.hstack((-I, D_x, np.zeros((n, m))))
        bottom_row = np.hstack((A, np.zeros((m, n)), np.zeros((m, m))))
        # print("middle row shape", top_row.shape, middle_row.shape, bottom_row.shape)
        # exit()
        block_matrix = np.vstack((top_row, middle_row, bottom_row))
        return block_matrix

    def set_zero_grad(self):
        self.model.zero_grad()
    
    def update_params(self, lr):
        with torch.no_grad():
            for param in self.model.parameters():
                param.data -= lr * param.grad.data
    
    def criterion(self, c_pred, x_target, Q=None):
        if self.config.task == "portfolio":
            x_pred = self.solver(c_pred[0], self.train_data, self.block_matrix, self.qp_solver, Q, self.hp_lamb)
        else:
            x_pred = self.solver(c_pred, self.train_data, self.block_matrix, self.qp_solver, Q, self.hp_lamb)
        return (((x_target - x_pred)**2).sum())/ x_target.shape[0]

    def inference(self, input, data):
        with torch.no_grad():
            return self.solver(self.model(input), data, None, 0.0)
    
    def get_lr_by_armijo(self, data, prev_lr):
        with torch.no_grad():
            sol, z = data["sol"], data["z"]
            z = torch.from_numpy(z).float()
            sol = torch.from_numpy(sol).float()

            func, params = make_functional(self.model)
                
            grad_norm = get_squared_grad_norm(self.model.parameters())
            base_lr = prev_lr
            base_loss = self.criterion(self.solver(func(params, z), data, self.block_matrix, self.hp_lamb), sol)

            lr = base_lr
            while True:
                for p1, p2 in zip(self.model.parameters(), params):
                    # p1.grad -= torch.mean(p1.grad)
                    p2.data = p1.data - lr*p1.grad
                
                loss = self.criterion(self.solver(func(params, z), data, self.block_matrix, self.hp_lamb), sol)
                if loss <= base_loss - 0.5*lr*grad_norm:
                    break
                if lr < 1e-10:
                    break
                lr *= 0.9
                # print("intermediate loss", loss,"lr", lr, "base_loss", base_loss, "grad_norm", grad_norm)
            return lr
    

    def get_loss(self):
        return self.criterion
    
    def get_solver(self):
        return self.lp_solver
    
    def forward(self, x):
        c_t = self.model(x)
        return c_t
    
    def get_target_fn(self):
        return self.get_target
        
    
    def get_target(self, c_pred, sol, Q=None):
        """ To ensure consistency with the other methods, we return the target as a tensor"""
        return torch.from_numpy(sol).float().to(c_pred.device)


class ShortestPath(torch.autograd.Function):
    """ Define a new function for shortest path to avoid the legacy autograd
    error """

    @staticmethod
    def forward(ctx, c_pred, data, block_matrix, solver, Q=None, gamma=None):
        ctx.data = data
        ctx.weights = c_pred.detach().cpu().numpy()
        ctx.grad_output_placeholder = np.zeros(2*ctx.data["A"].shape[1] +ctx.data["A"].shape[0])
        ctx.block_matrix = block_matrix
        ctx.save_for_backward(c_pred)
        ctx.Q = Q
        ctx.gamma = gamma

        ctx.suggested_tours, ctx.duals = get_dual_sol_from_LP_variables(solver, ctx.weights)
        # ctx.suggested_tours, ctx.duals = get_dual_sol_from_LP_variables(ctx.data["A"], ctx.data["b"], ctx.weights, gamma)
        return torch.from_numpy(ctx.suggested_tours).float().to(c_pred.device)


    @staticmethod
    def backward(ctx, grad_output):
        weights = ctx.saved_tensors
        Q_pred = ctx.Q
        gamma = ctx.gamma

        grad_output_numpy = grad_output.detach().cpu().numpy()
        assert grad_output_numpy.shape == ctx.suggested_tours.shape
        gradient = np.zeros_like(grad_output_numpy)
        n = ctx.data["A"].shape[1]

        for i in range(ctx.suggested_tours.shape[0]):
            block_matrix_old = ctx.block_matrix.copy()
            if Q_pred is not None:
                update_block_matrix(ctx.block_matrix, ctx.duals[i], ctx.suggested_tours[i], 2*gamma*Q_pred[i])
            else:
                update_block_matrix(ctx.block_matrix, ctx.duals[i], ctx.suggested_tours[i])
            ctx.grad_output_placeholder[:n] = grad_output_numpy[i]
            gradient[i] = -(np.linalg.pinv(ctx.block_matrix)@ctx.grad_output_placeholder)[:n]
        return torch.from_numpy(gradient).to(grad_output.device), None, None, None, None, None

def update_block_matrix(block_matrix, new_lamb, new_x, new_Q=None):
    n = new_lamb.shape[0]

    # Calculate the diagonal matrices of the new_lamb and new_x
    new_D_lamb = -np.diag(new_lamb)
    new_D_x = -np.diag(new_x)

    # Update the block matrix with the new D(lamb) and D(x)
    block_matrix[:n, n:2*n] = new_D_lamb
    block_matrix[n:2*n, n:2*n] = new_D_x
    if new_Q is not None:
        block_matrix[:n, :n] = new_Q
    # print("Shapes", block_matrix.shape, new_D_lamb.shape, new_D_x.shape, block_matrix[:n, n:2*n].shape)
    # exit()

