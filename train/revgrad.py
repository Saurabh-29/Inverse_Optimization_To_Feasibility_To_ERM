import torch
import torch.nn as nn 
import pickle
import numpy as np
from scipy.optimize import linprog
import sys
sys.path.append("../")
import Algorithm.LinearProgramMethod as lpm
from functorch import make_functional, grad, make_functional_with_buffers

from utils import get_squared_grad_norm, get_loss_with_LP_variables, get_test_loss
from data_utils import loadData
from models import get_model
from utils import *
import cvxpy as cp
import time
import Algorithm.LearningMethod as lm
from fractions import Fraction
from cvxpy import OSQP
from Solver import Solver

class RevGrad():
    def __init__(self, config):
        super(RevGrad, self).__init__()
        """
        args:
            config: configuration dict for the model
        
        Call the dataloader, model, optimizer, and other parameters from the config file.
        Define the states.
        """
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
        self.hp_lamb = self.config.method["hp_lamb"]

        self.optimizer_name, self.optimizer = return_optimizer_from_string(config["optimizer"], self.model)
        
        self.scheduler_flag = False
        try:
            if self.config.optimizer.lr_scheduler:
                T = self.config.train_params.total_steps
                decayRate = np.power(1.0/T, 1.0/T)
                self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)
                self.scheduler_flag = True
        except:
            print("lr scheduler not defined")

        self.train_data, self.valid_data, self.test_data = kernelize_data(config, self.train_data, self.valid_data, self.test_data)
        self.A = self.train_data["A"]
        self.b = self.train_data["b"]
        self.count= 0 
        self.c_fixed = None

        self.beta = 1.0
        self.T = 100
        self.iter = 0
        self.eta = 1.0
        self.step = 0

        lp_params = {"A": self.A, "b": self.b, "margin": self.config.method.eps, "problem": "LP", "task": config["task"], "gamma": self.hp_lamb}
        self.lp_solver = Solver(lp_params, cfg=config)

        qp_params = {"A": self.A, "b": self.b, "margin": self.config.method.eps, "problem": "QP_KKT", "task": config["task"], "gamma": self.hp_lamb}
        self.qp_solver = Solver(qp_params, cfg=config)

    def run_lr_scheduler(self):
        if self.scheduler_flag:
                self.my_lr_scheduler.step()

    def get_eta(self, c_target, c_pred, eta_max=2.0):
        """
        args:
            c_target: target cost vector (batch_size, num_var)
            c_pred: predicted cost vector (batch_size, num_var)
            eta_max: maximum eta value
        
        returns:
            eta_prop: eta value calculated armijo 


        This function assumes the loss used for grad is mean squared loss. Code is inspired from
        the surrogate github repo.
        """
        with torch.no_grad():
            eta_prop = eta_max
            grad = (c_pred - c_target)/len(c_pred)

            for _ in range(100):
                lhs = ((c_pred - eta_prop*grad - c_target)**2).sum() / len(c_pred)
                rhs = ((c_pred - c_target)**2).sum() /len(c_pred) - 0.5*eta_prop*(grad**2).sum()
                if lhs > rhs:
                    eta_prop = eta_prop*0.9
                elif (eta_prop) <= 1e-6:
                    return eta_prop
                else:
                    return eta_prop
        return eta_prop

    def trainer(self):
        """
            Trains the model for num_epochs, and saves the model and log data.
        """
        full_input = torch.from_numpy(self.train_data["z"]).float()
        prev_lr = self.config["optimizer"]["lr"]
        log_data = []
        batch_size = self.config.train_params.batch_size
        a2 = 0 #to measure time


        ## load the loss function accordingly
        if self.config.method.surrogate:
            loss_fn = self.get_surrogate_loss
        else:
            loss_fn = self.criterion_with_target

        for epoch in range(self.train_params["num_epochs"]):
            a1 = time.time()
            loss_sum = 0
            for iter_num in range(0, full_input.shape[0], batch_size):
                # print(iter_num, "iter_num")
                self.step += 1

                # load the input and solution for the random batch of size batch_size
                index = torch.randperm(full_input.shape[0])[0:batch_size]
                input = full_input[index].to(self.device)
                sol = self.train_data["sol"][index]
                
                # get the c_target for the batch of input
                with torch.no_grad():
                    c_t = self.model(input)
                    new_target = self.get_target_KKT(c_t.to(self.cpu).detach(), sol).to(self.device)
                    new_target = new_target.detach()
                    self.c_fixed = c_t.clone().detach()
                
                prev_lr = self.config["optimizer"]["lr"]

                # run the inner loop for k(m in SSO) steps
                for _ in range(self.config["optimizer"]["k"]):
                    c_t = self.model(input)

                    # for the surrogate loss, update eta
                    if self.config.method.surrogate:
                        self.eta = self.get_eta(new_target, c_t, eta_max=200.0)
                        #make it a state in the init
                        total_steps = int(Fraction(self.config["optimizer"]["total_steps"]))
                        if self.config.optimizer.eta_schedule == "exponential":
                            self.eta = self.eta*np.power(1/total_steps, self.step/total_steps)
                        elif self.config.method.eta_schedule == "stochastic":
                            self.eta = self.eta*(1/ np.sqrt(self.step))
                   
                    # get the loss and calculate the gradient
                    loss = loss_fn(new_target, c_t)
                    self.set_zero_grad()
                    loss.backward()
                    loss_sum += loss
                    
                    if self.optimizer_name == "exact":
                        """
                        for the exact method, we solve the linear regression exactly with matrix inverse
                             and update the parameters
                        """
                        new_input = torch.cat([input, torch.ones(input.shape[0], 1)], dim=1)
                        theta = torch.matmul(torch.linalg.pinv(torch.matmul(new_input.T, new_input)), new_input.T)
                        theta = torch.matmul(theta, new_target)
                        lr=0
                        self.model.model[0].weight = nn.Parameter(theta[:-1, :].T)
                        self.model.model[0].bias = nn.Parameter(theta[-1])
                        # loss = self.criterion_with_target(new_target, c_t)
                        self.count += 1
                    elif self.optimizer_name=="cg":
                        from cg_torch import compute_A_and_b, conjugate_gradient
                        X = torch.cat([input, torch.ones(input.shape[0], 1)], dim=1) #append 1 for bias term
                        y_hat = new_target
                        theta = self.get_theta()
                        A, b = compute_A_and_b(X, y_hat, c_t)
                        lr=0
                        theta = conjugate_gradient(A, b, theta, self.config.optimizer.m)
                        self.model.model[0].weight = nn.Parameter(theta[:-1, :].T)
                        self.model.model[0].bias = nn.Parameter(theta[-1])
                       
                    elif self.optimizer_name == "armijo":
                        lr = self.get_lr_by_armijo_with_target(input, new_target, prev_lr, loss_fn)
                        self.update_params(lr)
                        self.iter += 1
                        prev_lr = 1.5*lr
                    elif self.optimizer_name == "Armijo":
                        lr = self.optimizer.step(self.model, input, new_target, loss_fn)
                        # self.update_params(lr)
                        self.iter += 1
                        
                    else:
                        self.optimizer.step()
                        lr = self.config["optimizer"]["lr"]
                
                self.run_lr_scheduler()

            a2 += time.time() - a1
            # log the data
            log_data, train_loss = update_log_data(log_data, self.config, self.model, self.lp_solver, lr, loss_sum*batch_size/full_input.shape[0], epoch,
                                self.train_data, self.valid_data, self.test_data)
                                
            a2 += time.time() -a1
            print("Epoch: {}, {}, time: {}".format(epoch, train_loss, a2))
        
        # save the log data and model
        save_log_file(log_data, self.config, lr)
        save_model(self.model, self.optimizer, self.config, lr)


    def get_theta(self):
        w1 = self.model.model[0].weight.data.clone().T 
        w2 = self.model.model[0].bias.data.clone().unsqueeze(0)
        # print("shapes", w1.shape, w2.shape)
        theta = torch.cat([w1, w2], dim=0)
        return theta


    def get_loss(self, loss="MSE"):
        if loss == "MSE":
            return self.criterion_with_target
        else:
            return self.get_surrogate_loss
    
    def get_solver(self):
        return self.lp_solver
    
    def forward(self, x):
        return self.model(x)

    
    def get_target_fn(self):
        if self.config.method.KKT:
            if self.config.task ==  "portfolio":
                return self.get_target_KKT_QP
            else:
                return self.get_target_KKT
        else:
            return self.get_target_red_cost
    
    def load_ckpt(self, ckpt_path):
        """
        args:
            ckpt_path: path to the checkpoint file
        
        Loads the model and optimizer parameters from the checkpoint file.
        """
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def define_lp_and_solve(self, A, b):
        """ 
        args:
            A: constraint matrix of the form A_B^-1 A_N
            b: constraint vector (reduced cost)
        
        Solves the reduced cost LP  with minimizing the cost function ||c||_2^2
        returns:
            c: cost vector
        """
        (dim_constraints, dim_target) = A.shape
        x = cp.Variable(dim_target)
        constr = [A@x <= b]
        cost = cp.sum_squares(x)
        obj = cp.Minimize(cost)
        prob = cp.Problem(obj, constr)
        prob.solve(solver = cp.GUROBI)
        # prob.solve()
        return x.value
    
    def define_lp_and_solve_KKT(self, c, A, x_sol):
        """
        args:
            c: predicted cost vector
            A: constraint matrix
            x_sol: solution vector
        
        Solves the LP for KKT conditions with minimizing the cost function ||delta_c||_2^2

        Returns the delta_c vector
        """
        margin = self.config.method.eps
        x_dim = A.shape[1]
        y_dim = A.shape[0]
        lamb = cp.Variable(y_dim)
        mu = cp.Variable(x_dim)
        delta_c = cp.Variable(x_dim)
        solver_options = {'GUROBI': {'Presolve': 2, 'WarmStart': 2}}
        x_sol = x_sol>=1e-6
        if self.config.method.pos_tgt:
            """ predicted cost should be positive for pos_tgt"""
            # print("positive targetttttt")
            constr = [A.T@(lamb) -delta_c-c+ mu == 0.0, mu >= 0, mu[x_sol>=1e-6]==0.0, mu[x_sol<1e-6]>=margin, (delta_c+c) >= 0.0]
        else:
            # constr = [A.T@(lamb) -delta_c-c+ mu == 0.0, mu >= 0, mu[x_sol>=1e-6]==0.0, mu[x_sol<1e-6]>=margin]
            constr = [A.T@(lamb) -delta_c-c+ mu == 0.0, cp.multiply(x_sol, mu)==0.0,  mu+x_sol>=margin]
        obj = cp.Minimize(cp.sum((delta_c)**2))
        prob = cp.Problem(obj, constr)
        # prob.solve(solver = cp.GUROBI, verbose=False, solver_opts=solver_options)
        # prob.solve(solver=cp.GUROBI)
        prob.solve(solver=OSQP)
        return delta_c.value

    def get_target_KKT(self, c_pred, sol, Q=None):
        """
        args:
            c_pred: predicted cost vector (batch_size, num_var)
            sol: solution vector (batch_size, num_var)

        returns:
            c_target: target cost vector calculated by solving the KKT conditions (batch_size, num_var)
        """
        A = self.A
        with torch.no_grad():
            batch_size = c_pred.shape[0]
            c_target = torch.zeros((batch_size, c_pred.shape[1]))
            for i in range(batch_size):
                delta_c = self.qp_solver.solve_QP_KKT(c_pred[i].numpy(), sol[i])
                c_target[i] = torch.from_numpy(delta_c).float() + c_pred[i]
        return c_target
    
    def get_target_KKT_QP(self, c_pred, sol, Q_pred):
        """
        args:
            c_pred: predicted cost vector (batch_size, num_var)
            sol: solution vector (batch_size, num_var)

        returns:
            c_target: target cost vector calculated by solving the KKT conditions (batch_size, num_var)
        """
        A = self.A
        with torch.no_grad():
            batch_size = c_pred.shape[0]
            c_target = torch.zeros((batch_size, c_pred.shape[1]))
            if self.config.q_pred:
                q_target = torch.zeros((batch_size, Q_pred.shape[1], Q_pred.shape[2]))
                Q_pred = Q_pred.numpy()
            for i in range(batch_size):
                delta_c = self.qp_solver.solve_QP_KKT(c_pred[i].numpy(), sol[i], Q_pred[i])
                c_target[i] = torch.from_numpy(delta_c[0]).float() + c_pred[i]
                if self.config.q_pred:
                    q_target[i] = torch.from_numpy(delta_c[1]).float() + Q_pred[i]
        
        if self.config.q_pred:
            return [c_target, q_target]
        else:
            return [c_target, None]
    
    def get_target_red_cost(self, c_pred, basis, not_basis):
        """
        args:
            c_pred: predicted cost vector (batch_size, num_var)
            basis: basis index (batch_size, num_basis)
            not_basis: non basis matrix (batch_size, num_not_basis)

        returns:
            c_target: target cost vector calculated by solving the reduced cost LP (batch_size, num_var)
        """
        with torch.no_grad():
            num_var = basis.shape[1] + not_basis.shape[1]
            batch_size = basis.shape[0]
            c_target = np.zeros((batch_size, num_var))
            A_inv_grad, reduced_cost = get_A_inv_grad(c_pred.numpy(), self.A, basis, not_basis)
            reduced_cost -=self.config.method.eps
            for i in range(batch_size):
                c_target[i] = self.define_lp_and_solve(A_inv_grad[i], reduced_cost[i])
            
            c_target = torch.from_numpy(c_target).float() + c_pred
            return c_target

    def set_zero_grad(self):
        """
        Sets the gradient of the model parameters to zero.
        """
        self.model.zero_grad()

    def set_attr_c_fixed(self, c_fixed):
        """
        args:
            c_fixed: fixed cost vector
        
        Sets the c_fixed value.
        """
        self.c_fixed = c_fixed.detach()
    

    def set_attr_eta(self, eta):
        """
        args:
            eta: eta value
        
        Sets the eta value.
        """
        self.eta = eta

    def get_surrogate_loss(self, c_pred, c_target):
        """
        args:
            c_target: target cost vector (batch_size, num_var)
            c_pred: predicted cost vector (batch_size, num_var)
            eta: hyperparameter for the surrogate loss

        returns:
            loss: surrogate loss (scalar)

        Defines the surrogate loss for the model. From the paper, ignore the first term as 
        it is constant. The second term is the dot product of the target and predicted cost.
        The third term is the regularisation loss. 
        """
        eta = self.eta
        c_fixed = self.c_fixed
        loss_1 = 0.5*((c_target.detach() - c_fixed.detach())**2).sum()/ len(c_pred) 
        # loss_2 = torch.tensordot(c_fixed.detach()-c_target.detach(), c_pred) / len(c_pred)
        # loss_3 = (1/(2*eta))*((c_fixed.detach() - c_pred)**2).sum()

        loss_2 = torch.tensordot(c_fixed.detach()-c_target.detach(), c_pred-c_fixed.detach()) / len(c_pred)
        loss_3 = (1/(2*eta))*((c_fixed.detach() - c_pred)**2).sum() 
        loss = (loss_1.item() + loss_2 + loss_3) 
        # loss = (loss_1 + loss_2 + loss_3) 
        # print("loss 1, loss 2, loss 3",  loss_1.item(), loss_2.item(), loss_3.item(), eta)
        return loss
        # print("c_target", c_target)
        # exit(0)
        # return 0.5*((c_target - c_pred.detach())**2).sum() / c_target.shape[0] 

    def update_params(self, lr):
        """
        args:
            lr: learning rate
        Updates the model parameters by -lr*gradient
        """
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad.data
    
    def criterion_with_target(self, c_pred, c_target, Q=None):
        """
        args:
            c_target: target cost vector (batch_size, num_var)
            c_pred: predicted cost vector (batch_size, num_var)
            Q: just to ensure consistency bu won't be used
        
        returns:
            loss: mseloss loss (scalar)
        """
        if self.config.task=="portfolio":
            loss = ((c_target[0] - c_pred[0])**2).sum()
            if self.config.q_pred:
                loss +=  ((c_target[1] - c_pred[1])**2).sum()

            return 0.5*loss / c_target[0].shape[0]
        else:
            return 0.5*((c_target - c_pred)**2).sum() / c_target.shape[0] 
    
    def get_lr_by_armijo_with_target(self, input, target, prev_lr, loss_fn):
        """
        args:
            input: input tensor (batch_size, input_dim)
            target: target cost vector (batch_size, num_var)
            prev_lr: previous learning rate
            loss_fn: loss function to use
        
        returns:
            lr: learning rate calculated by armijo rule

        A different way to implement the armijo rule using torch functional module which makes model
        a function of the parameters thus the (new) loss can be calculated by 
        model(input, params- lr*grad).
        """
        with torch.no_grad():
            z = input
            # func, params = make_functional(self.model)
            func, params, buffer = make_functional_with_buffers(self.model)
            grad_norm = get_squared_grad_norm(self.model.parameters())
            base_lr = prev_lr
            base_loss = loss_fn(func(params, buffer, z), target)

            lr = base_lr
            while True:
                for p1, p2 in zip(self.model.parameters(), params):
                    if p1.grad is not None:
                        p2.data = p1.data - lr*p1.grad
                
                # loss = self.criterion_with_target(target, func(params, z))
                loss = loss_fn(func(params, buffer, z), target)
                c_armijo = self.config["optimizer"]["c"]
                if loss <= base_loss - c_armijo*lr*grad_norm:
                    break
                if lr < 1e-10:
                    break
                lr *= 0.9
            return lr
    