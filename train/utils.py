import numpy as np
import torch
import sys
sys.path.append("../")
# import Algorithm.LinearProgramMethod as lpm
import wandb
# import Algorithm.LearningMethod as lm
from armijo import Armijo
import time
# from analyse_pm import A_eq, b_eq


def update_log_data(log_data, config, model, solver, lr, loss, epoch,
                    train_data, val_data=None, test_data=None, qp_solver=None):
    """ 
    args:
        log_data: list of dictionaries containing the loss values
        config: configuration dict
        model: model object
        lr: learning rate
        loss: loss value
        epoch: current epoch
        train_data: train data
        val_data: validation data
        test_data: test data
    
    This function updates the log_data list with the current loss values.
    """
    if epoch%config.train_params.eval_every == 0:
        train_loss = get_test_loss(model, solver, train_data, "train", config, qp_solver)
        if val_data is not None:
            train_loss.update(get_test_loss(model, solver, val_data, "valid", config,  qp_solver))
        if test_data is not None:
            train_loss.update(get_test_loss(model, solver, test_data, "test", config, qp_solver))

        train_loss["grad_norm"] = np.sqrt(get_squared_grad_norm(model.parameters()))
        train_loss["lr"] = lr
        train_loss["loss"] = loss
        log_data.append(train_loss)
        if config["debug"]:
            pass
        else:
            log_loss_in_wandb(train_loss)
        return log_data, train_loss
    else:
        return log_data, None


def create_loss_dict(data_split):
    """
    args:
        data_split: train, valid, test
    
    This function creates an empty dictionary with the appropriate keys.
    """
    loss_dict = {}
    loss_dict["{}_l2_loss".format(data_split)] = 0
    loss_dict["{}_estimate_loss".format(data_split)] = 0
    loss_dict["{}_subopt_loss".format(data_split)] = 0
    loss_dict["{}_cost_ratio".format(data_split)] = 0
    loss_dict["{}_our_subopt_loss".format(data_split)] = 0
    return loss_dict


def get_exp_name(cfg):
    data_name = cfg.data_path.split("/")[-1].split(".")[0]
    exp_name = cfg.exp_name if "exp_name" in cfg.keys() else ""
    path_name = "{}_{}_{}_{}_{}_{}".format(cfg.method.name, cfg.optimizer["name"], np.around(cfg["optimizer"]["lr"],4), cfg["optimizer"]["k"], data_name, exp_name)
    return path_name

def save_log_file(log_data, cfg, lr):
    data_name = cfg.data_path.split("/")[-1].split(".")[0]
    exp_name = cfg.exp_name if "exp_name" in cfg.keys() else ""
    if optimizer == "armijo":
        lr = 0
    path_name = "log_data_{}_{}_{}_{}_{}_{}.npy".format(cfg.method.name, cfg.optimizer["name"], np.around(lr,4), cfg["optimizer"]["k"], data_name, exp_name)
    np.save(path_name, np.array(log_data))

def save_model(model, optimizer, cfg, lr, scheduler=None):
    data_name = cfg.data_path.split("/")[-1].split(".")[0]
    exp_name = cfg.exp_name if "exp_name" in cfg.keys() else ""
    if optimizer == "armijo":
        lr = 0
    path_name = "model_{}_{}_{}_{}_{}_{}.pt".format(cfg.method.name, cfg.optimizer["name"], np.around(lr,4), cfg["optimizer"]["k"], data_name, exp_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': cfg.train_params.num_epochs,
        'scheduler': scheduler.state_dict() if scheduler is not None else None
        }, path_name)
    print("Model saved at {}".format(path_name))

def load_ckpt(model, optimizer, ckpt_path):
    if ckpt_path == "None":
        return model, optimizer, 0

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        scheduler.load_state_dict(checkpoint['scheduler'])
    except:
        print("scheduler state dict not found")
    
    try :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("optimizer state dict not found")
    try:
        epoch = checkpoint['epoch']
    except:
        epoch = 0 

    return model, optimizer, epoch
    
def get_test_loss(model, solver, data, data_split, config, qp_solver=None):
    """
    args:
        model: model object
        data: data dictionary
        data_split: train, valid, test
    
    Runs the LP solver for data_split and returns the loss values.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        cpu = torch.device('cpu')
        loss_dict = create_loss_dict(data_split)
        l2_loss, _estimate_loss, _subopt_loss, _cost_ratio, _our_subopt_loss = 0, 0, 0, 0, 0
        sol, z, c = data["sol"], data["z"], data["c"]
        N = len(sol)
        max_batch_size = min(1000, N)

        for start_index in range(0, N, max_batch_size):
            s, t = start_index, start_index+max_batch_size
            sol, z, c = data["sol"][s:t], data["z"][s:t], data["c"][s:t]
            if config.task == "portfolio":
                Q = data["Q"][s:t]
            else:
                Q = None

            c_pred = model(torch.from_numpy(z).float().to(device))

            if config.task == "portfolio":
                c_t = c_pred[0].to(cpu).numpy()
                if config.q_pred:
                    Q_pred = c_pred[1].to(cpu).numpy()
            else:
                c_t = c_pred.to(cpu).numpy()

            if config.q_pred:
                sol_pred = get_sol_from_LP_variables(solver, c_t, Q_pred)
            else:
                sol_pred = get_sol_from_LP_variables(solver, c_t, Q)

            c_target = np.zeros_like(c_t)
            
            if qp_solver is not None:
                bs = c_target.shape[0]
                for i in range(bs):
                    delta_c = qp_solver.solve_QP_KKT(c_t[i], sol[i])
                    c_target[i] = delta_c + c_t[i]

            if config.task == "portfolio":
                diff = sol_pred-sol
                _subopt_loss += np.sum(-c_t*diff) 

                extra_sum = np.einsum('bj, bjj, bj->b', diff, Q, diff)
                """already c is negative in the data so no need for negative """
                _estimate_loss += np.sum(c*diff) + 0.1*np.sum(extra_sum)
            else:
                _subopt_loss += np.sum(c_t*(sol-sol_pred))
                _estimate_loss += np.sum(c*(sol_pred-sol))

            l2_loss += np.sum((sol_pred-sol)**2)
            _cost_ratio += np.sum(np.sum(c*sol_pred, axis=1)/np.sum(c*sol, axis=1))

        loss_dict["{}_subopt_loss".format(data_split)] += _subopt_loss/ N
        loss_dict["{}_l2_loss".format(data_split)] += l2_loss/ N
        loss_dict["{}_estimate_loss".format(data_split)] += _estimate_loss/ N
        loss_dict["{}_cost_ratio".format(data_split)] += _cost_ratio/ N
        # loss_dict["{}_our_subopt_loss".format(data_split)] += _our_subopt_loss/ N
        
        return loss_dict

def get_squared_grad_norm(parameters):
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

def get_loss_with_LP_variables(solver, c_t, sol):
    """
    args:
        A, b, c: LP variables
        c_t: model output
        sol: ground truth
    
    returns: loss value (sol_pred-sol)**2
    """
    sol_pred = get_sol_from_LP_variables(solver, c_t)
    loss = np.sum((sol_pred-sol)**2)
    return loss

def get_sol_from_LP_variables(solver, c, Q=None):
    """
    args:
        A, b, c: LP variables
    """
    sol_pred = np.zeros_like(c)
    for i in range(c.shape[0]):
        if Q is not None:
            x_pred = solver.solve_LP(c[i], Q[i])
        else:
            x_pred = solver.solve_LP(c[i])
        sol_pred[i] = x_pred
    return sol_pred

def get_dual_sol_from_LP_variables(solver, c):
    """
    args:
        A, b, c: LP variables
    
    returns: solution and dual variables

    Mainly used for the QPTL method.
    """
    sol_pred = np.zeros_like(c)
    dual_pred = np.zeros_like(c)
    for i in range(c.shape[0]):
        x_pred, lamb_dual = solver.solve_LP(c[i], return_dual=True)
        sol_pred[i] = x_pred
        dual_pred[i] = lamb_dual
    return sol_pred, dual_pred

def get_dual_sol_from_LP_variables_old(A, b, c, gamma):
    """
    args:
        A, b, c: LP variables
        gamma: regularization parameter
    
    returns: solution and dual variables

    Mainly used for the Optnet method.
    """
    solver = lpm.Solver(A, b, c, gamma)
    sol_pred = np.zeros_like(c)
    dual_pred = np.zeros_like(c)
    for i in range(c.shape[0]):
        _, lamb_dual, x_pred = solver.ComputeLP_dual(i)
        sol_pred[i] = x_pred
        dual_pred[i] = lamb_dual
    return sol_pred, dual_pred

def log_loss_in_wandb(loss_dict):
    loss_dict["grad_norm"] = np.log(loss_dict["grad_norm"])
    wandb.log(loss_dict)


optimizer = {   "SGD": torch.optim.SGD, 
                "adam": torch.optim.Adam,
                "adagrad": torch.optim.Adagrad,
                "exact":  torch.optim.SGD,
                "cg": torch.optim.SGD,
                "SSO_ada": torch.optim.Adagrad,
                "SSO_sgd": torch.optim.SGD,
            }

def return_optimizer_from_string(cfg, model):
    """
    args:
        cfg: configuration dict
        model: model object
    
    returns: optimizer object
    """
    if cfg["name"] == "armijo":
        return cfg["name"], None
    elif cfg["name"] in ["Armijo", "SSO"]:
        return cfg["name"], Armijo(model, {"lr": cfg["lr"], "b_bwd": cfg["b_bwd"], "b_fwd": cfg["b_fwd"], "c": cfg["c"], "m": cfg["m"]})
    else:
        return cfg["name"], optimizer[cfg["name"]](model.parameters(), lr=cfg["lr"])

def kernelize_data(config, train_data, valid_data=None, test_data=None):
    if config.dataset.poly_kernel:
        degree = config.dataset.poly_degree
        gamma = config.dataset.poly_gamma
        if test_data is not None:
            test_data["z"] = lm.PolyKernel(z = test_data["z"], benchmark_z = train_data["z"], gamma=gamma, degree=degree, coef0=1.0)
        if valid_data is not None:
            valid_data["z"] = lm.PolyKernel(z = valid_data["z"], benchmark_z = train_data["z"], gamma=gamma, degree=degree, coef0=1.0)
        train_data["z"] = lm.PolyKernel(z = train_data["z"], benchmark_z = train_data["z"], gamma=gamma, degree=degree, coef0=1.0)
    elif config.dataset.exp_kernel:
        gamma = config.dataset.exp_gamma
        if test_data is not None:
            test_data["z"] = lm.ExpKernel(z = test_data["z"], benchmark_z = train_data["z"], gamma=gamma)
        if valid_data is not None:
            valid_data["z"] = lm.ExpKernel(z = valid_data["z"], benchmark_z = train_data["z"], gamma=gamma)
        train_data["z"] = lm.ExpKernel(z = train_data["z"], benchmark_z = train_data["z"], gamma=gamma)

    return train_data, valid_data, test_data
        

# @profile
def get_A_inv_grad(c_pred, A, basis, non_basis):
    A = A.astype('float64')
    # print(A.shape, basis.shape, non_basis.shape)
    A_b = A[:, basis].transpose(1, 0, 2)
    A_n = A[:, non_basis].transpose(1, 0, 2)
    # print(A_b.shape, A_n.shape)
    # exit()
    A_b_inv = np.linalg.pinv(A_b, rcond=1e-10)

    A_inv_grad = np.zeros((A_b.shape[0], non_basis.shape[1], A.shape[1]))
    # results = np.einsum('Bbn,Bnk ->Bbk', A_b_inv, A_n).transpose(0, 2, 1)

    results = np.matmul(A_b_inv, A_n).transpose(0, 2, 1)

    # print("diff", np.sum((results-results_2)**2))
    # exit()

    batch_size = c_pred.shape[0]
    for i in range(basis.shape[0]):
        A_inv_grad[i][:, basis[i]] = results[i]
        A_inv_grad[i][np.arange(len(non_basis[i])), non_basis[i]] = -1
    return A_inv_grad, 0
    # reduced_cost = -(A_inv_grad @ c_pred.reshape(batch_size, -1, 1)).reshape(batch_size, -1)
    c_n = c_pred[:, non_basis][np.arange(batch_size), np.arange(batch_size), :]
    c_b = c_pred[:, basis][np.arange(batch_size), np.arange(batch_size), :]
    # multiplier = np.einsum('Bbk,Bkn ->Bbn', A_b_inv, A_b)
    reduced_cost = c_n - np.einsum('Bb,Bbk,Bkn  ->Bn', c_b, A_b_inv, A_n)

    # print(reduced_cost, "numpy")
    return A_inv_grad,  reduced_cost



def clean_sol(sol):
    bs = sol.shape[0]

    for i in range(bs):
        sum = 0
        for j in range(sol.shape[1]):
            sum += sol[i][j]
            if sum>200:
                sol[i][j]=0
    return sol
            

if __name__ == "__main__":
    A = np.random.randint(2, size=(200, 1000))
    A = A.astype('float64')
    basis_len = 200
    non_basis_len = 800
    total_len = basis_len + non_basis_len

    A_inv_grad = np.zeros((basis_len, non_basis_len, A.shape[1]))

    N = 10
    BS = 100
    a1 = time.time()
    for i in range(N):
        sol = np.random.randint(2, size=(BS, 1000))
        sol = clean_sol(sol)
        
        basis = np.where(sol>1e-6)[0].reshape(BS, -1)
        not_basis = np.where(sol<=1e-6)[0].reshape(BS, -1)
        c = np.random.randint(10, size=(BS, 1000))
        A_inv_grad, reduced_cost = get_A_inv_grad(c, A, basis, not_basis)
    
    print("time taken", time.time()-a1)
    # print(A_inv_grad.shape, reduced_cost.shape)


      
    