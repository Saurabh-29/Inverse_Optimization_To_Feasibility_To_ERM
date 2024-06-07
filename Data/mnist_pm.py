import numpy as np
import cvxpy as cp
import pickle
import Algorithm.LinearProgramMethod as lpm


def get_basis_for_data(A, b, c):
    basic, non_basic, solution = lpm.ComputeBasis(c=c, A=A, b=b)
    return basic, non_basic, solution

def put_things_in_dict(a, z, c, A, b, bas, n_bas, sol):
    a['c'] = c
    a['A'] = A
    a['b'] = b
    a['basis'] = bas
    a['not_basis'] = n_bas
    a['sol'] = sol
    a['z'] = z
    return a


##set these parameters
dim = 12
limit_train = 10
limit_val = 10



num_vertex = dim*dim
num_edges = 2*dim*(dim-1)
A_eq = np.zeros((num_vertex, num_edges))
b_eq = np.ones((num_vertex))


def e_2_v(x1, y1, x2, y2):
    if x1<0 or x2<0 or y1<0 or y2<0:
        return -1
    if x1>=dim or x2>=dim or y1>=dim or y2>=dim:
        return -1
    return x*(dim)+y

edge_2_id = {}
id_2_edge = {}
edge_list = []

cnt = 0
for i in range(dim):
    for j in range(dim):
        if i == dim-1:
            pass
        else:
            edge_2_id[(i,j,i+1,j)] = cnt
            id_2_edge[cnt] = (i,j,i+1,j)
            edge_list.append((i*dim+j, (i+1)*dim+j))
            cnt += 1
        
        if j == dim-1:
            pass
        else:
            edge_2_id[(i,j,i,j+1)] = cnt
            id_2_edge[cnt] = (i,j,i,j+1)
            edge_list.append((i*dim+j, (i)*dim+j+1))
            cnt += 1

for i in range(dim):
    for j in range(dim):
        vertex_num = i*dim+j

        try: 
            e1 = edge_2_id[(i,j,i+1,j)]
            A_eq[vertex_num, e1] = 1
        except:
            pass
        try:
            e2 = edge_2_id[(i,j,i,j+1)]
            A_eq[vertex_num, e2] = 1
        except: 
            pass
        try:
            e3 = edge_2_id[(i-1,j,i,j)]
            A_eq[vertex_num, e3] = 1
        except:
            pass
        try:
            e4 = edge_2_id[(i,j-1,i,j)]
            A_eq[vertex_num, e4] = 1
        except:
            pass


M = np.zeros((num_vertex, num_edges))

for i in range(num_edges):
    e = id_2_edge[i]
    x1, y1, x2, y2 = e
    v1 = x1*(dim)+y1
    v2 = x2*(dim)+y2
    M[v1, i] = 10
    M[v2, i] = 1


def solve_LP(c):
    x = cp.Variable(num_edges)
    constr = [A_eq@x == b_eq, x >= 0]
    obj = cp.Minimize(c.T@x)
    prob = cp.Problem(obj, constr)
    prob.solve()
    # print(x.value)
    # print((x.value-sol_train[i]).sum())
    return x.value

def get_z_c(data_path, split="train", limit=None):
    z = np.load(data_path+split+"_full_images.npy")
    c = np.load(data_path+split+"_board.npy")
    sol = np.load(data_path+split+"_perfect_matching.npy")

    c = c.reshape(-1, num_vertex)
    c_new = np.matmul(c, M)

    if limit is not None:
        z = z[:limit]
        c_new = c_new[:limit]
        sol = sol[:limit]
    
    for i in range(c_new.shape[0]):
        sol[i] = solve_LP(c_new[i])

    return z, c_new, sol


data_path = "../dataset/mnist_matching/{}x{}_mnist_nonunique/".format(dim, dim)

z_train, c_train, sol_train = get_z_c(data_path, split="train", limit=limit_train)
z_val, c_val, sol_val = get_z_c(data_path, split="val", limit=limit_val)
z_test, c_test, sol_test = get_z_c(data_path, split="test", limit=limit_val)

basic_valid, nonb_valid, solution_valid = get_basis_for_data(A_eq, b_eq, c_val)
basic_test, nonb_test, solution_test = get_basis_for_data(A_eq, b_eq, c_test)
basic_train, nonb_train, solution_train = get_basis_for_data(A_eq, b_eq, c_train)


dict_data = {}
dict_data["config"] = {}
dict_train, dict_valid, dict_test = {}, {}, {}
dict_train = put_things_in_dict(dict_train, z_train, c_train, A_eq, b_eq, basic_train, nonb_train, sol_train)
dict_valid = put_things_in_dict(dict_valid, z_val, c_val, A_eq, b_eq, basic_valid, nonb_valid, sol_val)
dict_test = put_things_in_dict(dict_test, z_test, c_test, A_eq, b_eq, basic_test, nonb_test, sol_test)

dict_data["train"] = dict_train
dict_data["valid"] = dict_valid
dict_data["test"] = dict_test
dict_data["ver_2_edge"] = M
dict_data["edges_list"] = edge_list

##save in pickle file
name = data_path + "pm_{}_mini_basis_{}".format(dim, limit_train)
with open('{}.pkl'.format(name), 'wb') as f:
    print("Saving to file")
    pickle.dump(dict_data, f)

###Save M to file as npy
# np.save("dataset/mnist_matching/12x12_mnist_nonunique/ver_2_edge_dim_{}.npy".format(dim), M)

##solve the Perfect matching

##for train set
for i in range(limit_train):
    x = cp.Variable(num_edges)
    constr = [A_eq@x == b_eq, x >= 0]
    obj = cp.Minimize(c_train[i].T@x)
    prob = cp.Problem(obj, constr)
    prob.solve()
    # print(x.value)
    print((x.value-sol_train[i]).sum())
    print("b_eq", np.matmul(A_eq, x.value))
    print("b_eq_sol", np.matmul(A_eq, sol_train[i]))
    print(x.value-sol_train[i])
    print(np.abs((x.value-sol_train[i])).sum())
    exit()
