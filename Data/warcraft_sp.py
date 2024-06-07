import sys
sys.path.append("./")
sys.path.append("../")


import numpy as np
import Algorithm.DataGeneration as dg
import Algorithm.LinearProgramMethod as lpm
import hydra
import torch
import cvxpy as cp
import pickle
from scipy.optimize import linprog

from prototype import *

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


def get_A_b(dim):
    edges = edges_from_grid(dim, neighbourhood_fn="8-grid")
    edges_list = edges_to_list_pair(edges, dim)
    num_vertex = dim*dim
    num_edges = len(edges_list) 
    A = np.zeros((num_vertex, 2*num_edges), dtype=int)
    b = np.zeros((num_vertex))
   

    # print(edges_list, len(edges_list))
    edges_list = sorted(edges_list)
    # print(edges_list)

    for i, e in enumerate(edges_list):
        x,y = e
        A[x, i] = 1
        A[y, i] = -1
        A[x, num_edges+i] = -1
        A[y, num_edges+i] = 1
    
    b[0]=1
    b[-1]=-1

    return A, b, edges_list

def get_basis_for_data(A, b, c):
    basic, non_basic, solution = lpm.ComputeBasis(c=c, A=A, b=b)
    return basic, non_basic, solution


def generate_dataset_sp(limit_train=None, limit_val=None):
    # if limit is None:
    #     limit = 10000000 #infinity
    data_folder = "./dataset/warcraft_shortest_path_oneskin/18x18/"
    z_train, c_train, A_train, b_train, edges_list = convert_to_edge_weights(data_folder, split="train", limit=limit_train)
    z_valid, c_valid, A_valid, b_valid, edges_list = convert_to_edge_weights(data_folder, split="val", limit=limit_val)
    z_test, c_test, A_test, b_test, edges_list = convert_to_edge_weights(data_folder, split="test", limit=limit_val)

    
    print("Shapes", z_train.shape, c_train.shape, A_train.shape, b_train.shape)
    basic_valid, nonb_valid, solution_valid = get_basis_for_data(A_valid, b_valid, c_valid)
    basic_test, nonb_test, solution_test = get_basis_for_data(A_test, b_test, c_test)
    basic_train, nonb_train, solution_train = get_basis_for_data(A_train, b_train, c_train)
 
    dict_data = {}
    dict_data["config"] = {}
    dict_train, dict_valid, dict_test = {}, {}, {}
    dict_train = put_things_in_dict(dict_train, z_train, c_train, A_train, b_train, basic_train, nonb_train, solution_train)   
    dict_valid = put_things_in_dict(dict_valid, z_valid, c_valid, A_valid, b_valid, basic_valid, nonb_valid, solution_valid)
    dict_test = put_things_in_dict(dict_test, z_test, c_test, A_test, b_test, basic_test, nonb_test, solution_test)
    dict_data["train"] = dict_train
    dict_data["valid"] = dict_valid
    dict_data["test"] = dict_test
    dict_data["edges_list"] = edges_list

    print(dict_data.keys())
    name = data_folder + "mini_{}".format(limit_train)

    with open('{}.pkl'.format(name), 'wb') as f:
        print("Saving to file")
        pickle.dump(dict_data, f)

def get_solution_by_KKT(A, b, c, x_sol, basis=None, not_basis=None):

    x_dim = A.shape[1]
    y_dim = A.shape[0]

    lamb = cp.Variable(y_dim)
    mu = cp.Variable(x_dim)
    c_tar = cp.Variable(x_dim)
    constr = [A.T@(lamb) -c_tar + mu == 0, mu >= 0, mu[x_sol>1e-6]==0, mu[x_sol<=1e-6]>=1]
    obj = cp.Minimize(cp.sum(c_tar**2))
    prob = cp.Problem(obj, constr)
    prob.solve(eps=1e-6)
    
    return c_tar.value
   
           
def convert_to_edge_weights(data_folder, split="train", limit=None):
    # data_folder="./dataset/warcraft_shortest_path_oneskin/18x18/"
    try:
        z = np.load(data_folder+"{}_maps.npy".format(split))[:limit]
        c = np.load(data_folder+"{}_vertex_weights.npy".format(split))[:limit]
    except:
        z0 = np.load(data_folder+"{}_maps_part0.npy".format(split))
        c0 = np.load(data_folder+"{}_vertex_weights_part0.npy".format(split))

        z1 = np.load(data_folder+"{}_maps_part1.npy".format(split))
        c1 = np.load(data_folder+"{}_vertex_weights_part1.npy".format(split))

        z = np.concatenate((z0, z1), axis=0)[:limit]
        c = np.concatenate((c0, c1), axis=0)[:limit]

        print("Loaded from npy files")
    a_s, b_s, edges_list  = get_A_b(c.shape[-1])
    # sol = np.load(data_folder+"{}_shortest_paths.npy".format(split))

    # A = np.zeros((z.shape[0], a_s.shape[0], a_s.shape[1]))
    # b = np.zeros((z.shape[0], b_s.shape[0]))
    # for i in range(z.shape[0]):
    #     A[i] = a_s
    #     b[i] = b_s
    # print(c.shape, A.shape, b.shape, sol.shape, z.shape)

    c_new = np.zeros((z.shape[0], a_s.shape[1]))

    c = c.reshape(z.shape[0], -1)
    num_edges = len(edges_list)
    for i, e in enumerate(edges_list):
        x, y = e
        c_new[:, i] = c[:, x] #+0e-4*np.random.uniform(0, 1.0, size=(z.shape[0]))
        c_new[:, num_edges+i] = c[:, y] #+0e-4*np.random.uniform(0, 1.0, size=(z.shape[0]))
    print("Shape", c_new.shape, z.shape)
    return z, c_new, a_s, b_s, edges_list

def ComputeLP(A, b, c):
    (dim_constraints, dim_target) = A.shape
    x = cp.Variable(dim_target)
    constr = [A@x == b, x >= 0]
    obj = cp.Minimize(c.T@x)
    prob = cp.Problem(obj, constr)
    prob.solve()
    # prob.solve()
    return x.value, prob.value, constr[0].dual_value

def get_basis_from_sol(A, b, c, sol=None):
    dim_target = c.shape[0]
    x = cp.Variable(dim_target)
    constr = [A@x == b, x >= 0]
    obj = cp.Minimize(c.T@x)
    prob = cp.Problem(obj, constr)
    prob.solve()

    print("constraint:", c.shape, (constr[0].dual_value).shape)
    reduced_cost = c + (A).T @ constr[0].dual_value
    dim_target, dim_constraints = A.shape[1], A.shape[0]
    idx = np.argpartition(reduced_cost, -(dim_target - dim_constraints))
    basic_tmp = idx[:dim_constraints]
    nonb_tmp = idx[dim_constraints:]
    basic = np.sort(basic_tmp)
    nonb = np.sort(nonb_tmp)

    del prob
    return basic, nonb, x.value

if __name__ == "__main__":
#    convert_to_edge_weights()
   generate_dataset_sp(limit_train=10000, limit_val=1000)