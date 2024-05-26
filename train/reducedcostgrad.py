import torch
import torch.nn as nn 
import pickle
import numpy as np
from scipy.optimize import linprog
import sys
sys.path.append("../")
import Algorithm.LinearProgramMethod as lpm
from functorch import make_functional, grad

from utils import get_squared_grad_norm, get_loss_with_LP_variables, get_test_loss
from data_utils import loadData
from models import get_model
from utils import *
import time 
from Solver import Solver


class ReducedCostGradient():
    def __init__(self, config):
        super(ReducedCostGradient, self).__init__()
        self.model_params = config["model_params"]
        self.device = torch.device('cuda' if (torch.cuda.is_available() and  config["device"]=="cuda")  else 'cpu')
        # self.device = torch.device('cpu')
        self.cpu = torch.device('cpu')
        self.model = get_model(config["model_params"]).to(self.device)
        self.config = config
        self.data_path = config["data_path"]
        self.data = loadData(self.data_path, cfg=config)

        self.train_params = config["train_params"]

        self.train_data = self.data.train_data
        self.valid_data = self.data.valid_data
        self.test_data = self.data.test_data

        self.optimizer_name, self.optimizer = return_optimizer_from_string(config["optimizer"], self.model)
        

        self.train_data, self.valid_data, self.test_data = kernelize_data(config, self.train_data, self.valid_data, self.test_data)
        self.A = self.train_data["A"]
        self.b = self.train_data["b"]
        lp_params = {"A": self.A, "b": self.b, "problem": "LP"}
        self.lp_solver = Solver(lp_params)
        
    def trainer(self):
        input = torch.from_numpy(self.train_data["z"]).float()
        prev_lr = self.config["optimizer"]["lr"]
        log_data = []
        a2 = 0
        for epoch in range(self.train_params["num_epochs"]):
            a1 = time.time()
            c_t = self.model(input)
            # loss = self.criterion(c_t, self.train_data["A_b_inv"], self.train_data["A_n"], self.train_data["basis"], 
            #                         self.train_data["not_basis"])
            loss = self.criterion(c_t, self.train_data["basis"], self.train_data["not_basis"])

            self.set_zero_grad()
            loss.backward()

            if self.optimizer_name == "armijo":
                lr = self.get_lr_by_armijo(self.train_data, prev_lr)
                self.update_params(lr)
                prev_lr = 1.5*lr
            else:
                self.optimizer.step()
                lr = self.config["optimizer"]["lr"]

            log_data, train_loss = update_log_data(log_data, self.config, self.model, self.lp_solver, lr, loss, epoch,
                                self.train_data, self.valid_data, self.test_data)
                                
            a2 += time.time() -a1
            print("Epoch: {}, {}, time: {}".format(epoch, train_loss, a2))

        save_log_file(log_data, self.config, lr)
        save_model(self.model, self.optimizer, self.config, lr)
    

    def get_corresponding_sol(self, x, basis):
        x_sol = np.zeros((x.shape[0], basis.shape[1]))
        for i in range(x_sol.shape[0]):
            x_sol[i,:] = x[i, basis[i]]
        return x_sol
    

    # def criterion_sol(self, c_pred, sol):
    def criterion_sol(self, c_pred, sol, basis_full=None, not_basis_full=None):
        batch_size = c_pred.shape[0]
        loss = 0
        # for i in range(batch_size):
        #     continue
        #     if self.config.method.x_basis:
        #         basis = np.where(sol[i]>1e-6)[0].reshape(1, -1)
        #         not_basis = np.where(sol[i]<=1e-6)[0].reshape(1, -1)
        #     else:
        #         basis = basis_full[i].reshape(1, -1)
        #         not_basis = not_basis_full[i].reshape(1, -1)

            # A_inv_grad, _ = get_A_inv_grad(c_pred[i].detach().numpy().reshape(1,-1), self.A, basis, not_basis)
            # A_inv_grad = torch.from_numpy(A_inv_grad).float().squeeze()
            # reduced_cost = -(torch.matmul(A_inv_grad, c_pred[i].view(-1, 1))).view(1, -1)
            # loss += (1-reduced_cost).clamp(min=0).sum()
        
        bs = batch_size
        A_inv_grad, _ = get_A_inv_grad(c_pred.detach().numpy().reshape(bs,-1), self.A, basis_full, not_basis_full)
        A_inv_grad = torch.from_numpy(A_inv_grad).float().squeeze()
        # print("A_inv_grad", A_inv_grad.shape, c_pred.shape)
        reduced_cost = -(torch.bmm(A_inv_grad, c_pred.unsqueeze(2))).view(bs, -1)
        loss += (1-reduced_cost).clamp(min=0).sum()

        return loss.sum()/batch_size


    def criterion(self, c_pred, basis, not_basis):
        batch_size = c_pred.shape[0]
        # print("batch_size", batch_size, basis.shape, not_basis.shape)
        # exit()
        A_inv_grad, _ = get_A_inv_grad(c_pred.detach().numpy(), self.A, basis, not_basis)
        A_inv_grad = torch.from_numpy(A_inv_grad).float()
        c_pred_reshaped = c_pred.view(batch_size, -1, 1)
        reduced_cost = -(torch.matmul(A_inv_grad, c_pred_reshaped)).view(batch_size, -1)
        reduced_cost =(1-reduced_cost).clamp(min=0)
        return reduced_cost.sum()/batch_size    

    def criterion_old(self, c_pred, A_b_inv, A_n, basis, not_basis):
        c_n = torch.gather(c_pred, 1, torch.from_numpy(not_basis).long())
        c_b = torch.gather(c_pred, 1, torch.from_numpy(basis).long())

        batch_size = c_pred.shape[0]
        reduced_cost = (c_n.reshape(batch_size, 1, -1) - torch.bmm(c_b.reshape(batch_size, 1,-1), 
                            torch.bmm(A_b_inv, A_n))).reshape(batch_size, -1)

        if self.config.method.l2:
            reduced_cost = (1-reduced_cost)**2
        else:
            reduced_cost = (1-reduced_cost).clamp(min=0)
            if self.config.method.l2_1:
                reduced_cost = reduced_cost**2
            if self.config.method.huber:
                reduced_cost =  self.huber_loss(reduced_cost)

        
        reduced_cost = reduced_cost.sum()/batch_size
        # print("costs", new_cost, reduced_cost)
        return reduced_cost
    
    def huber_loss(self, value):
        return torch.where(torch.abs(value)<1, 0.5*value**2, torch.abs(value)-0.5)
    
    def get_loss(self):
        return self.criterion_sol
    
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

    def get_lr_by_armijo(self, data_split, prev_lr):
        with torch.no_grad():
            lr = prev_lr
            z = torch.from_numpy(data_split["z"]).float()
            func, params = make_functional(self.model)
            grad_norm = get_squared_grad_norm(self.model.parameters())
            # base_loss = self.criterion(func(params,z), data_split["A_b_inv"], data_split["A_n"], data_split["basis"], 
            #                             data_split["not_basis"])
            base_loss = self.criterion(func(params,z), data_split["basis"], 
                                        data_split["not_basis"])
            while True:
                for p1, p2 in zip(self.model.parameters(), params):
                    p2.data = p1.data - lr*p1.grad
                
                # loss = self.criterion(func(params,z), data_split["A_b_inv"], data_split["A_n"], data_split["basis"], 
                #                             data_split["not_basis"])
                loss = self.criterion(func(params,z),data_split["basis"], 
                                            data_split["not_basis"])
                if loss <= base_loss - 0.5*lr*grad_norm:
                    break
                if lr < 1e-10:
                    break
                # print("intermediate loss", loss,"lr", lr, "base_loss", base_loss, "grad_norm", grad_norm)
                lr = lr*0.9
                
            return lr
    
    def update_params(self, lr):
        with torch.no_grad():
            for p in self.model.parameters():
                p.data -= lr*p.grad
    def set_zero_grad(self):
        self.model.zero_grad()