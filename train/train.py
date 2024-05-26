import torch
import torch.nn as nn 
import pickle
import numpy as np
from scipy.optimize import linprog
import sys
sys.path.append("../")
import Algorithm.LinearProgramMethod as lpm
from functorch import make_functional, grad

from reducedcostgrad import ReducedCostGradient
from revgrad import RevGrad
# from naive_sol import NaiveSol
from blackbox import Blackbox
from identity import Identity
from optnet import OptNet
import hydra
import wandb
from utils import *
import omegaconf
import time 
from SPOplus import SPOplus
from fractions import Fraction
from datetime import datetime
from Solver import Solver

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR

method_name = {  "redcost": ReducedCostGradient, 
            "revgrad": RevGrad,
            # "naivesol": NaiveSol, 
            "blackbox": Blackbox, 
            "identity": Identity,
            "optnet": OptNet,
            "SPOplus": SPOplus,
            }

def get_eta(c_target, c_pred, eta_max=2.0):
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


def get_eta_scheduler(cfg, eta, iter_num):
    eta_schedule = cfg.optimizer.eta_schedule
    if eta_schedule == "constant":
        return eta
    if eta_schedule == "exponential":
        total_steps = int(Fraction(cfg["optimizer"]["total_steps"]))
        return eta*np.power(1/total_steps, iter_num/total_steps)
    if eta_schedule == "stochastic":
        return eta*(1/ np.sqrt(iter_num))
    else:
        raise NotImplementedError("eta_type {} not implemented".format(cfg.method.eta_type))


def get_scheduler(cfg, optimizer):
    if cfg.optimizer.lr_scheduler:
        if cfg.optimizer.scheduler_name == "step":
            return MultiStepLR(optimizer, milestones=cfg.optimizer.milestones, gamma=cfg.optimizer.scheduler_gamma), True
        else:
            print("Scheduler {} not implemented".format(cfg.optimizer.scheduler_name))
    return None, False

# @profile
def trainer(cfg, method):
    """
    args:
        cfg: configuration dict given by hydra 
    """
    optimizer = method.optimizer
    optimizer_name = cfg.optimizer.name
    time_list = []

    if optimizer_name in  ["SSO", "SSO_ada", "SSO_sgd"]:
        loss_fn = method.get_loss(loss="SSO")
    else:
        loss_fn = method.get_loss()

    solver = method.get_solver()
    get_target_fn = method.get_target_fn()
    device = torch.device('cuda' if (torch.cuda.is_available() and  cfg["device"]=="cuda")  else 'cpu')
    method.model.to(device)

    full_input = torch.from_numpy(method.train_data["z"]).float()
    # print("full input shape", full_input.shape)
    # exit()
    batch_size = cfg.train_params.batch_size

    log_data = []
    lr = cfg.optimizer.lr
    start_lr = cfg.optimizer.lr
    loss_sum = 0
    iter_num = 0

    lr_scheduler, scheduler_flag = get_scheduler(cfg, optimizer)  

    epoch_cnt, steps_cnt = 0, 0 
    method.model, optimizer, _ = load_ckpt(method.model, optimizer, cfg.ckpt_path)


    ##define the QP get_target to get our sub-opt loss
    qp_params = {"A": method.train_data["A"], "b": method.train_data["b"], "margin": 1.0, "problem": "QP_KKT", "task": cfg["task"]}
    qp_solver_subopt = None ##Solver(qp_params)

    for epoch in range(cfg.train_params.num_epochs): #epochs
        a1 = time.time()
        loss_sum = 0
        for i in range(0, full_input.shape[0], batch_size): #iterations per epoch = dataset/batch-size
            index = torch.randperm(full_input.shape[0])[0:batch_size]
            z = full_input[index].to(device)
            sol = method.train_data["sol"][index]
            if cfg.task == "portfolio":
                Q = method.train_data["Q"][index]
            

            c_pred = method.forward(z)
            if cfg.task == "portfolio":
                if cfg.q_pred:
                    target = get_target_fn(c_pred[0], sol, c_pred[1])
                else:
                    target = get_target_fn(c_pred[0], sol, Q)
                    
            else:
                target = get_target_fn(c_pred, sol)
            
            if optimizer_name in  ["SSO", "SSO_ada", "SSO_sgd"]:
                with torch.no_grad():
                    target = target.detach()
                    c_fixed = c_pred.clone().detach() 
                    method.set_attr_c_fixed(c_fixed)  
            
            if cfg.method.name == "redcost":
                if cfg.method.x_basis:
                    pass
                else:
                    basis = method.train_data["basis"][index]
                    not_basis = method.train_data["not_basis"][index]

            optimizer.zero_grad()
            for j in range(cfg.optimizer.k): #k steps of gradient descent per target calculation
                c_pred = method.forward(z)
                
                if optimizer_name in  ["SSO", "SSO_ada", "SSO_sgd"]:
                    eta_val = batch_size #get_eta(target, c_pred, eta_max=1000.0)
                    eta = get_eta_scheduler(cfg, eta_val, iter_num)
                    method.set_attr_eta(eta)  
                    # print("eta is"s, eta) 

                if cfg.method.name == "SPOplus":
                    c_gt = method.train_data["c"][index]
                    loss = loss_fn(c_pred, c_gt, target)
                elif cfg.method.name == "redcost":
                    if cfg.method.x_basis:
                        loss = loss_fn(c_pred, target)
                    else:
                        loss = loss_fn(c_pred, target, basis, not_basis)

                else:
                    if cfg.task == "portfolio":
                        if cfg.q_pred:
                            loss = loss_fn(c_pred, target, c_pred[1])
                        else:
                            loss = loss_fn(c_pred, target, Q)
                            # print("i am calculaating this loss")
                    else:
                        loss = loss_fn(c_pred, target)
                
                loss_sum += loss.item()
                print("Loss is", loss.item())

                optimizer.zero_grad()
                loss.backward()
               
                if cfg.optimizer.name in ["Armijo", "SSO"]:
                    optimizer.step(method.model, z, target, loss_fn)
                else:
                    optimizer.step()
                
            iter_num += 1
            steps_cnt += 1
        a2 = time.time() - a1
        time_list.append(a2)

        epoch_cnt += 1
        if scheduler_flag:
            lr_scheduler.step()
            lr = lr_scheduler._last_lr
        
        # print("wtffff is happening herere")
        log_data, train_loss = update_log_data(log_data, cfg, method.model, solver, lr, loss_sum*batch_size/full_input.shape[0], epoch,
                            method.train_data, method.valid_data, None, None)
                            
        print("Epoch: {}, {}, time: {}".format(epoch, train_loss, a2))

    save_log_file(log_data, cfg, start_lr)
    save_model(method.model, optimizer, cfg, start_lr)
    print("Average time per epoch: {}, method: {}, data: {}".format(np.mean(time_list), cfg.method.name, cfg.data_path))
    print("Time taken is :", time_list)
           

@hydra.main(version_base=None, config_path="../config", config_name="config")
def train_model(cfg):
    """
    args: 
        cfg: configuration dict given by hydra
        config_path: path to the configuration folder
        config_name: name of the configuration file
    
    this function taked the cfg file and calls the appropriate method (revgrad etc.)
    """

    print(cfg)
    debug = int(cfg.debug)
    if debug:
        pass
    else:
        run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}" + "_" + get_exp_name(cfg)
        wandb.init(project="Blackbox Non-convex", mode=cfg.wandb_mode, name=run_name)
        wandb.config.update(omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True))
        wandb.run.name = get_exp_name(cfg)
    method = method_name[cfg.method.name](cfg)
    if cfg.trainer == "general":
        trainer(cfg, method)
    elif cfg.trainer == "specific":
        method.trainer()
    else:
        raise NotImplementedError("Trainer {} not implemented".format(cfg.trainer))
    
if __name__ == "__main__":
    train_model()


        
