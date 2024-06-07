import sys
sys.path.append("./")
sys.path.append("../")


import numpy as np
import hydra
import torch
import cvxpy as cp
import pickle
from scipy.optimize import linprog
import DataGeneration as dg
import LinearProgramMethod as lpm

def put_things_in_dict(a, z, c, A, b, bas, n_bas, sol):
    a['c'] = c
    a['A'] = A
    a['b'] = b
    a['basis'] = bas
    a['not_basis'] = n_bas
    a['sol'] = sol
    a['z'] = z
    return a


def gen_shortest_path(cfg):
    dim_features = cfg.dim_features
    dim_edge_hori = cfg.dim_edge_hori
    dim_edge_vert = cfg.dim_edge_vert
    degree = cfg.degree
    additive_noise = cfg.additive_noise
    scale_noise_uni = cfg.scale_noise_uni
    scale_noise_div = cfg.scale_noise_div
    attack_threshold = cfg.attack_threshold
    attack_power = cfg.attack_power
    N_train = cfg.N_train
    N_valid = cfg.N_valid
    N_test = cfg.N_test
    
    dim_cost = dim_edge_hori * (dim_edge_vert + 1) + (dim_edge_hori + 1) * dim_edge_vert
    Coeff_Mat = np.random.binomial(n=1, p=0.5, size = (dim_cost, dim_features))


    z_train, c_train, A_train, b_train = dg.GenerateShortestPath(N_samples = N_train, dim_features = dim_features, Coeff_Mat=Coeff_Mat,
                                                                dim_edge_vert = dim_edge_vert, dim_edge_hori = dim_edge_hori,
                                                                degree=degree, additive_noise=additive_noise, scale_noise_uni=scale_noise_uni, scale_noise_div=scale_noise_div, attack_threshold=attack_threshold, attack_power=attack_power)
    z_valid, c_valid, A_valid, b_valid = dg.GenerateShortestPath(N_samples = N_valid, dim_features = dim_features, Coeff_Mat=Coeff_Mat,
                                                                dim_edge_vert = dim_edge_vert, dim_edge_hori = dim_edge_hori,
                                                                degree=degree, additive_noise=additive_noise, scale_noise_uni=scale_noise_uni, scale_noise_div=scale_noise_div, attack_threshold=attack_threshold, attack_power=attack_power)
    z_test, c_test, A_test, b_test = dg.GenerateShortestPath(N_samples = N_test, dim_features = dim_features, Coeff_Mat=Coeff_Mat,
                                                                dim_edge_vert = dim_edge_vert, dim_edge_hori = dim_edge_hori,
                                                                degree=degree, additive_noise=additive_noise, scale_noise_uni=scale_noise_uni, scale_noise_div=scale_noise_div, attack_threshold=attack_threshold, attack_power=attack_power)

    print(cp.installed_solvers())
    # ignoring the rest of the values of A as it is same for all the same
    A_train, A_valid, A_test = A_train[0, :], A_valid[0, :], A_test[0, :]
    b_train, b_valid, b_test = b_train[0, :], b_valid[0, :], b_test[0, :]

    basic_train, nonb_train, solution_train = lpm.ComputeBasis(c=c_train, A=A_train, b=b_train)
    basic_valid, nonb_valid, solution_valid = lpm.ComputeBasis(c=c_valid, A=A_valid, b=b_valid)
    basic_test, nonb_test, solution_test = lpm.ComputeBasis(c=c_test, A=A_test, b=b_test)
    print("Train: ", basic_train.shape, nonb_train.shape, solution_train.shape)
    print("Valid: ", basic_valid.shape, nonb_valid.shape, solution_valid.shape)
    print("Test: ", basic_test.shape, nonb_test.shape, solution_test.shape)
    dict_data = {}
    dict_data["config"] = cfg
    dict_train, dict_valid, dict_test = {}, {}, {}
    dict_train = put_things_in_dict(dict_train, z_train, c_train, A_train, b_train, basic_train, nonb_train, solution_train)
    dict_valid = put_things_in_dict(dict_valid, z_valid, c_valid, A_valid, b_valid, basic_valid, nonb_valid, solution_valid)
    dict_test = put_things_in_dict(dict_test, z_test, c_test, A_test, b_test, basic_test, nonb_test, solution_test)
    dict_data["train"] = dict_train
    dict_data["valid"] = dict_valid
    dict_data["test"] = dict_test

    print(dict_data.keys())
    name = cfg.name
    with open('{}.pkl'.format(name), 'wb') as f:
        pickle.dump(dict_data, f)
    with open('{}.pkl'.format(name), 'rb') as f:
        loaded_dict = pickle.load(f)

    print(loaded_dict.keys())
    print(loaded_dict["config"])
    print((loaded_dict["train"]["A"] == dict_data["train"]["A"]).all())

def generate_knapsack(cfg):
    dim_features = cfg.dim_features
    dim_edge_hori = cfg.dim_decision
    degree = cfg.degree
    additive_noise = cfg.additive_noise
    scale_noise_uni = cfg.scale_noise_uni
    scale_noise_div = cfg.scale_noise_div
    attack_threshold = cfg.attack_threshold
    attack_power = cfg.attack_power
    N_train = cfg.N_train
    N_valid = cfg.N_valid
    N_test = cfg.N_test
    
    dim_decision = cfg.dim_decision
    price = np.random.randint(low=1, high=1000, size=dim_decision)
    lower = np.amax(price)
    upper = (np.random.rand()-1)*lower + np.sum(price)
    Budget = [(upper-lower)*np.random.rand() + lower]
    Coeff_Mat = np.random.binomial(n=1, p=0.5, size = (dim_decision, dim_features))
    Theta_true = -np.concatenate((Coeff_Mat, np.zeros((dim_decision+1, dim_features))), axis=0) 
    
    z_train, c_train, A_train, b_train = dg.GenerateFractionalKnapsack(N_samples=N_train, dim_features=dim_features, dim_decision=dim_decision, Coeff_Mat=Coeff_Mat, price=price, Budget=Budget,
                                        degree=degree, additive_noise=additive_noise, scale_noise_uni=scale_noise_uni, scale_noise_div=scale_noise_div, attack_threshold = attack_threshold, attack_power = attack_power)
    z_valid, c_valid, A_valid, b_valid = dg.GenerateFractionalKnapsack(N_samples=N_valid, dim_features=dim_features, dim_decision=dim_decision, Coeff_Mat=Coeff_Mat, price=price, Budget=Budget,
                                        degree=degree, additive_noise=additive_noise, scale_noise_uni=scale_noise_uni, scale_noise_div=scale_noise_div, attack_threshold = attack_threshold, attack_power = attack_power)
    z_test, c_test, A_test, b_test = dg.GenerateFractionalKnapsack(N_samples=N_test, dim_features=dim_features, dim_decision=dim_decision, Coeff_Mat=Coeff_Mat, price=price, Budget=Budget,
                                        degree=degree, additive_noise=additive_noise, scale_noise_uni=scale_noise_uni, scale_noise_div=scale_noise_div, attack_threshold = attack_threshold, attack_power = attack_power)

    
    print(cp.installed_solvers())
    # ignoring the rest of the values of A as it is same for all the same
    A_train, A_valid, A_test = A_train[0, :], A_valid[0, :], A_test[0, :]
    b_train, b_valid, b_test = b_train[0, :], b_valid[0, :], b_test[0, :]
    # exit()
    basic_train, nonb_train, solution_train = lpm.ComputeBasis(c=c_train, A=A_train, b=b_train)
    basic_valid, nonb_valid, solution_valid = lpm.ComputeBasis(c=c_valid, A=A_valid, b=b_valid)
    basic_test, nonb_test, solution_test = lpm.ComputeBasis(c=c_test, A=A_test, b=b_test)
    print("Train: ", basic_train.shape, nonb_train.shape, solution_train.shape)
    print("Valid: ", basic_valid.shape, nonb_valid.shape, solution_valid.shape)
    print("Test: ", basic_test.shape, nonb_test.shape, solution_test.shape)
    dict_data = {}
    dict_data["config"] = cfg
    dict_train, dict_valid, dict_test = {}, {}, {}
    dict_train = put_things_in_dict(dict_train, z_train, c_train, A_train, b_train, basic_train, nonb_train, solution_train)
    dict_valid = put_things_in_dict(dict_valid, z_valid, c_valid, A_valid, b_valid, basic_valid, nonb_valid, solution_valid)
    dict_test = put_things_in_dict(dict_test, z_test, c_test, A_test, b_test, basic_test, nonb_test, solution_test)
    dict_data["train"] = dict_train
    dict_data["valid"] = dict_valid
    dict_data["test"] = dict_test

    print(dict_data.keys())
    name = cfg.name
    with open('{}.pkl'.format(name), 'wb') as f:
        pickle.dump(dict_data, f)
    with open('{}.pkl'.format(name), 'rb') as f:
        loaded_dict = pickle.load(f)

    print(loaded_dict.keys())
    print(loaded_dict["config"])
    print((loaded_dict["train"]["A"] == dict_data["train"]["A"]).all())

    
@hydra.main(version_base=None, config_path="../config", config_name="configDataGen")
def generator(cfg):
    # print("name", cfg.dataGen.name)
    # exit()

    if cfg.dataGen.name == "sp":
        gen_shortest_path(cfg.dataGen)
        print("Generated shortest path data")
    elif cfg.dataGen.name == "knapsack":
        generate_knapsack(cfg.dataGen)
        print("Generated fractional knapsack data")
    else: 
        print("Unknown dataset name")
        exit()


def test_my_idea():
    with open('dataset_1.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    cnt = 0
    for i in range(loaded_dict["train"]["A"].shape[0]):
        try:
            np.linalg.inv(loaded_dict["train"]["A"][i][:, loaded_dict["train"]["basis"][i]])
            cnt = i
            break
        except:
            # cnt += 1
            print("index {} is not invertible".format(i))
        
    index = cnt
    # print("cnt = ", cnt)
    # exit()
    A = loaded_dict["train"]["A"][index]
    b = loaded_dict["train"]["b"][index]
    c_real = loaded_dict["train"]["c"][index]
    basis = loaded_dict["train"]["basis"][index]
    not_basis = loaded_dict["train"]["not_basis"][index]
    sol = loaded_dict["train"]["sol"][index]
    # print(A.shape, b.shape, c.shape)

    x = linprog(c_real, A_eq=A, b_eq=b, bounds=(0, None))
    # print(np.rint(x.x))
    print(np.rint(sol))
    # print((np.rint(x.x) == np.rint(sol)).all())

    
    A_basis = A[:, basis] 
    # print(A_basis.shape)
    A_basis_inv = np.linalg.inv(A_basis)
    A_not_basis = A[:, not_basis]
    # print(basis, not_basis)
    A_inv_ineq = np.zeros((not_basis.shape[0], A.shape[1]+1))
    b_inv_ineq_grad = np.zeros((not_basis.shape[0]))
    
    # print(A_basis_inv.shape, A_not_basis.shape, A_inv_ineq.shape)
    A_inv_ineq[:, basis] = np.matmul(A_not_basis.T, A_basis_inv.T) 
    for i, index in enumerate(not_basis):
        A_inv_ineq[i, index] = -1
    
    A_inv_ineq[:, -1] = 1 #for margin

    ##to test the gradient of the solution
    c_new = np.random.randint(1, 100, (sol.shape[0]))
    # c_new = c_real
    c_new = 10*c_new/ np.sum(c_new)
    c_b_new = c_new[basis]
    c_n_new = c_new[not_basis]

    b_inv_ineq_grad = (c_n_new.reshape(1, -1) - np.matmul(c_b_new.reshape(1, -1), np.matmul(A_basis_inv, A_not_basis))).reshape(-1)

    # for i, index in enumerate(not_basis):
    #     print("Reduced cost for index (N)", index, c_new[index]- np.dot(c_b_new, np.matmul(A_basis_inv, A[:, index])))
    
    # for i, index in enumerate(basis):
    #     print("Reduced cost for index (B)", index, c_new[index]- np.dot(c_b_new, np.matmul(A_basis_inv, A[:, index])))

    print(b_inv_ineq_grad)
    # print(np.matmul(A_basis_inv, A_basis).shape, np.linalg.matrix_rank(A_basis))
    # exit(0)

    c_inv_grad = np.ones((A.shape[1]+1))
    c_inv_grad[-1] = -10 #maximize margin
    result_for_grad = linprog(c_inv_grad, A_ub=A_inv_ineq, b_ub=b_inv_ineq_grad)
    # print(result_for_grad.x)

    c_new_grad = c_new + result_for_grad.x[:-1]

    x_new = linprog(c_new_grad, A_eq=A, b_eq=b, bounds=(0, None))
    print("new solution: ", np.rint(x_new.x))
    print((np.rint(x_new.x) == np.rint(sol)).all())

def test_diff_loss_by_basis():
    with open('./dataset/dataset_1.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    A = loaded_dict["train"]["A"][0]
    b = loaded_dict["train"]["b"][0]
    c = loaded_dict["train"]["c"][0]
    sol = loaded_dict["train"]["sol"][0]
    basis = loaded_dict["train"]["basis"][0]
    not_basis = loaded_dict["train"]["not_basis"][0]


    c_pred = np.random.rand(c.shape[0])

    for epoch in range(5):
        basis_pred, nb_pred, sol_pred = get_basis_from_sol(A, b, c_pred, sol)

        # print("basis:", basis, basis_pred)

        c_b_new = c_pred[basis_pred]
        c_n_new = c_pred[nb_pred]

        A_basis = A[:, basis_pred] 
        # print(A_basis.shape)
        A_basis_inv = np.linalg.inv(A_basis)
        A_not_basis = A[:, nb_pred]

        A_inv_ineq = np.zeros((nb_pred.shape[0], A.shape[1]))
        b_inv_ineq_grad_analyse = (c_n_new.reshape(1, -1) - np.matmul(c_b_new.reshape(1, -1), np.matmul(A_basis_inv, A_not_basis))).reshape(-1)
        b_inv_ineq_grad = np.ones((nb_pred.shape[0]))

        A_inv_ineq[:, basis] = -np.matmul(A_not_basis.T, A_basis_inv.T) 
        for i, index in enumerate(nb_pred):
            A_inv_ineq[i, index] = 1

        cnt=0
        # print(b_inv_ineq_grad)
        nb_list = []
        for i, bb in enumerate(nb_pred):
            # if bb  not in basis:
            if sol[bb] == 0:
                cnt+=1
                # print(b, cnt)
                A_inv_ineq[i] = 0
                b_inv_ineq_grad[i] = 0
                b_inv_ineq_grad_analyse[i] = 0
                nb_list.append(bb)
        np_list = np.array(nb_list)

        print("max tau", np.max(A_inv_ineq), np.sum(b_inv_ineq_grad_analyse))
        dim_target = c.shape[0]
        x = cp.Variable(dim_target)
        constr = [A_inv_ineq@x <= -100*b_inv_ineq_grad]
        obj = cp.Minimize(cp.sum(((c_pred-x)**2)))
        prob = cp.Problem(obj, constr)
        prob.solve(solver = cp.GUROBI)
        # print("new solution: ", np.abs(x.value).sum())

        c_pred = x.value

        x_new = linprog(c_pred, A_eq=A, b_eq=b, bounds=(0, None))

        # print("new solution: ", np.rint(x_new.x))
        # print("old solution: ", np.rint(sol))
        print("Epoch", epoch, (np.rint(x_new.x) == np.rint(sol)).all(), ((np.rint(x_new.x)- np.rint(sol))**2).sum(), ((np.rint(sol_pred)- np.rint(sol))**2).sum())
        # print("Epoch", epoch, x_new[nb_list], sol[nb_list])

def get_basis_from_sol(A, b, c, sol):
    dim_target = c.shape[0]
    x = cp.Variable(dim_target)
    constr = [A@x == b, x >= 0]
    obj = cp.Minimize(c.T@x)
    prob = cp.Problem(obj, constr)
    prob.solve(solver = cp.GUROBI)

    # print("constraint:",constr[0].dual_value, constr[1].dual_value)
    reduced_cost = c + (A).T @ constr[0].dual_value
    dim_target, dim_constraints = A.shape[1], A.shape[0]
    idx = np.argpartition(reduced_cost, -(dim_target - dim_constraints))
    basic_tmp = idx[:dim_constraints]
    nonb_tmp = idx[dim_constraints:]
    basic = np.sort(basic_tmp)
    nonb = np.sort(nonb_tmp)

    return basic, nonb, x.value

def get_basis_with_random_selection(path="temp", num_of_perms=1):
    with open ('{}.pkl'.format(path), 'rb') as f:
        loaded_dict = pickle.load(f)
    basis =  loaded_dict["train"]["basis"]
    sol = loaded_dict["train"]["sol"]
    not_basis = loaded_dict["train"]["not_basis"]

    # print(basis.shape, sol.shape, not_basis.shape)
    # print(basis[0], np.where(sol[0]==1), not_basis[0])

    num_features = sol.shape[1]
    num_samples = basis.shape[0]
    num_basis = basis.shape[1]

    all_indexes = np.arange(sol.shape[1], dtype=np.int32)

    for j in range(num_of_perms):
        for i in range(num_samples):
            probs = np.ones(num_features)
            probs[np.where(sol[i]>1e-4)[0]] = 0 
            probs = probs/np.sum(probs)
            num_ones = len(np.where(sol[i]>1e-4)[0])
            flag=1
            if num_ones == num_basis:
                continue
            else:
                print("faillllll")
                print(sol[i], num_ones)
                exit()
            while(flag):
                random_sel = np.random.choice(all_indexes, num_basis-num_ones,  p=probs, replace=False)
                
                basic = sorted(np.concatenate((np.where(sol[i]==1)[0], random_sel)))
                print(i, num_basis, basic)
                non_b = np.setdiff1d(all_indexes, basic)
                # if np.linalg.cond(loaded_dict["train"]["A"][i][:, basic]) < 100:
                if np.linalg.matrix_rank(loaded_dict["train"]["A"][i][:, basic]) == num_basis:
                    print("condition number", np.linalg.cond(loaded_dict["train"]["A"][i][:, basic]) )
                    flag=0
                else:
                    print("Faillllllll", i, loaded_dict["train"]["basis"][i], probs[16:19], np.linalg.matrix_rank(loaded_dict["train"]["A"][i][:, basic]))
                    # exit()
                
                if random_sel[0]==17:
                    print("wtffffffffffffffffff")
                    print(basis[i], basic, num_ones, np.where(sol[i]==1)[0])
                    exit()
            loaded_dict["train"]["basis"][i] = basic
            loaded_dict["train"]["not_basis"][i] = non_b
            print("basis:", len(non_b), len(basic))
        exit()
        data_name = "{}_new_basis_perm_{}".format(path, j)
        with open('{}.pkl'.format(data_name), 'wb') as f:
            pickle.dump(loaded_dict, f)

def get_solution_by_KKT(A, b, c, x_sol, basis=None, not_basis=None):

    x_dim = A.shape[1]
    y_dim = A.shape[0]

    red_cost = np.linalg.inv(A[:,basis])@A[:, not_basis]
    # print(red_cost.shape, c.shape, basis.shape, not_basis.shape)
    red_cost_val = c[not_basis] - np.matmul(c[basis], red_cost)
    # print(red_cost_val)
    # exit()
    c_rc = cp.Variable(x_dim)
    constr_0 = [c_rc[not_basis]- c_rc[basis]*red_cost + red_cost_val >=0.0001]
    obj_0 = cp.Minimize(cp.sum((c_rc)**2))
    prob_0 = cp.Problem(obj_0, constr_0)
    prob_0.solve(solver = cp.GUROBI)
    # print("solution: ", c_rc.value)
    # exit()

   
    lamb = cp.Variable(y_dim)
    mu = cp.Variable(x_dim)
    c_tar = cp.Variable(x_dim)
    constr = [A.T@lamb - 0.99*c_tar -c+ mu == 0, mu >= 0, mu*x_sol==0]
    obj = cp.Minimize(cp.sum(c_tar**2))
    prob = cp.Problem(obj, constr)
    prob.solve(solver = cp.GUROBI)
    # print("solution: ", c_tar.value)

    for c_target  in [c_rc.value, c_tar.value]:
        x = cp.Variable(x_dim)
        constr_2 = [A@x == b, x >= 0]
        obj_2 = cp.Minimize(c.T@x+c_target.T@x)
        prob_2 = cp.Problem(obj_2, constr_2)
        prob_2.solve(solver = cp.GUROBI)

        print("solution: ", (x.value==x_sol).all(), np.sum(c_target**2))
        print("solution: ", x.value, x_sol)
    
       
def test_basis_by_perturb():
    # with open('./dataset/dataset_1.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    with open('./dataset/knapsack.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    batch_size = loaded_dict["train"]["A"].shape[0]

    print("old", loaded_dict["train"]["basis"][0])
    print("old", loaded_dict["train"]["not_basis"][0])

    red_cost = []
    for i in range(batch_size):
        A = loaded_dict["train"]["A"][i]
        b = loaded_dict["train"]["b"][i]
        basis =  loaded_dict["train"]["basis"][i]
        not_basis = loaded_dict["train"]["not_basis"][i]

        c_to_send = loaded_dict["train"]["c"][i] + np.random.rand(loaded_dict["train"]["c"][i].shape[0])*10
        get_solution_by_KKT(A, b, c_to_send, loaded_dict["train"]["sol"][i], basis, not_basis)
        exit()


        c = -loaded_dict["train"]["sol"][i]
        # c = 8*loaded_dict["train"]["c"][i] / np.sum(loaded_dict["train"]["c"][i])
        eps = (0.1+np.random.rand(c.shape[0])*0.2)*np.ones((c.shape[0]), dtype=np.float32)

        margin = np.matmul(A, eps)

        b_new = b  #+ margin

        # ans = linprog(c, A_eq=A, b_eq=b_new)
        dim_target = c.shape[0]
        x = cp.Variable(dim_target)
        constr = [A@x == b_new, x >= 0]
        obj = cp.Minimize(c.T@x)
        prob = cp.Problem(obj, constr)
        prob.solve(solver = cp.GUROBI)

        # print("constraint:",constr[0].dual_value, constr[1].dual_value)
        reduced_cost = c + (A).T @ constr[0].dual_value
        dim_target, dim_constraints = A.shape[1], A.shape[0]
        idx = np.argpartition(reduced_cost, -(dim_target - dim_constraints))
        basic_tmp = idx[:dim_constraints]
        nonb_tmp = idx[dim_constraints:]
        basic = np.sort(basic_tmp)
        nonb = np.sort(nonb_tmp)
        # print("basis:", basic, reduced_cost[nonb])
        red_cost.append(reduced_cost[nonb])
        # print("nonbasis:", nonb)

        sol_new = x.value
        basis_new = np.where(sol_new>0)[0]
        not_basis_new = np.where(sol_new<=0)[0]
       
        loaded_dict["train"]["basis"][i] = basic
        loaded_dict["train"]["not_basis"][i] = nonb
    print("new", loaded_dict["train"]["basis"][0])
    print("new", loaded_dict["train"]["not_basis"][0])
    loaded_dict["train"]["red_cost"] = np.array(red_cost)
    name = "dataset_1_new_basis"
    with open('{}.pkl'.format(name), 'wb') as f:
        pickle.dump(loaded_dict, f)

def test_eqivalence():
    data_1, data_2 = {}, {}
    with open('dataset_1.pkl', 'rb') as f:
        data_1 = pickle.load(f)
    with open('dataset_1_old_basis_w_red_cost.pkl', 'rb') as f:
        data_2 = pickle.load(f)
    
    for keys in data_1.keys():
        # print("keys: ", keys)
        if keys in ["train", "test", "valid"]:
            for keys_2 in data_1[keys].keys():
                print(keys, keys_2, (data_1[keys][keys_2]==data_2[keys][keys_2]).all())

if __name__ == "__main__":
    generator()
    # test_my_idea()
    # test_basis_by_perturb()

    # test_eqivalence()
    # test_diff_loss_by_basis()
    # get_basis_with_random_selection(path="./dataset/knapsack", num_of_perms=5)