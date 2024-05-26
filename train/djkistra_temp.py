import os
import pickle
import numpy as np
import heapq
import time 
import cvxpy as cp
from cvxpy import OSQP

class Djikstra():
    def __init__(self, edges_list, dim=12):
        super(Djikstra, self).__init__()

        self.num_vertex = dim*dim
        self.adjacency_matrix = np.zeros((self.num_vertex,8), dtype=np.int32)

        e_cnt = np.zeros((self.num_vertex,1), dtype=np.int32)
        self.edge_cost_map = {} # map from (u,v) to edge index to get the cost of the edge u,v
        self.num_edges = len(edges_list)

        for i, edges in enumerate(edges_list):
            u, v = edges[0], edges[1]
            
            self.adjacency_matrix[u][e_cnt[u]] = v
            e_cnt[u] += 1
            self.adjacency_matrix[v][e_cnt[v]] = u
            e_cnt[v] += 1

            self.edge_cost_map[(u,v)] = i
            self.edge_cost_map[(v,u)] = i+self.num_edges

        self.prev_transition = np.zeros((self.num_vertex,1), dtype=np.int32)
        self.visited = np.zeros((self.num_vertex,1), dtype=np.int32)
        self.sp = np.zeros((2*self.num_edges,1), dtype=np.int32)
    

    def forward(self, c):
        self.prev_transition *= 0 #vector
        self.sp *= 0 #vector
        self.visited *= 0 #vector
        
        priority_queue = [(1000000, (0,0))]
        curr_vertex = 0
        cur_cost = 0
        self.visited[0] = 1
        while self.visited[self.num_vertex-1]==0:
            for i in range(8):
                if self.adjacency_matrix[curr_vertex][i] == 0:
                    continue
                v = self.adjacency_matrix[curr_vertex][i]

                if self.visited[v] == 1:
                    continue    
                else:
                    heapq.heappush(priority_queue, (cur_cost+ c[self.edge_cost_map[(curr_vertex, v)]], (curr_vertex, v)))
            flag=1
            while(flag):
                cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
                if self.visited[cur_y] == 1:
                    continue
                else:
                    flag=0
                    break
            curr_vertex = cur_y
            self.prev_transition[cur_y] = cur_x
            self.visited[curr_vertex] = 1

        curr_vertex = self.num_vertex-1
        while curr_vertex!=0:
            u, v = self.prev_transition[curr_vertex][0], curr_vertex
            self.sp[self.edge_cost_map[(u, v)]] = 1
            curr_vertex = u
            # print(u, v)
        # print((data["train"]["sol"][index]-self.sp).sum())


data_path="../dataset/warcraft_shortest_path_oneskin/12x12/12x12_mini_200.pkl"

#read data
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

djikstra = Djikstra(data['edges_list'])

for i in range(1):
    """ run this to warm up the cache """
    print(i)
    for index in range(200):
        djikstra.forward(data["train"]["c"][index])

n_times = 1
a1 = time.time()
for i in range(n_times):
    print(i)
    for index in range(200):
        djikstra.forward(data["train"]["c"][index])

a2 = time.time()-a1 
 
from utils import get_sol_from_LP_variables

A = data["train"]["A"]
b = data["train"]["b"]
(dim_constraints, dim_target) = A.shape
x = cp.Variable(dim_target)
constr = [A@x == b, x >= 0, x<=1]
c = cp.Parameter(dim_target)
obj = cp.Minimize(c.T@x)
prob = cp.Problem(obj, constr)

a3 = time.time()
previous_solution = np.zeros((dim_target,1))
for i in range(n_times):
    for index in range(200):
        c.value = data["train"]["c"][index]
        prob.solve(warm_start=True)
        previous_solution = x.value

    # get_sol_from_LP_variables(data["train"]["A"], data["train"]["b"], data["train"]["c"])

a4 = time.time()-a3

print("Time spent on djikstra: {}, LP: {}".format(a2, a4))



def define_lp_and_solve_KKT(c, A, x_sol):
        """
        args:
            c: predicted cost vector
            A: constraint matrix
            x_sol: solution vector
        
        Solves the LP for KKT conditions with minimizing the cost function ||delta_c||_2^2

        Returns the delta_c vector
        """
        margin = 1
        x_dim = A.shape[1]
        y_dim = A.shape[0]
        lamb = cp.Variable(y_dim)
        mu = cp.Variable(x_dim)
        delta_c = cp.Variable(x_dim)
        solver_options = {'GUROBI': {'Presolve': 2, 'WarmStart': 2}}
        constr = [A.T@(lamb) -delta_c-c+ mu == 0.0, mu >= 0, mu[x_sol>=1e-6]==0.0, mu[x_sol<1e-6]>=margin]
        obj = cp.Minimize(cp.sum((delta_c)**2))
        prob = cp.Problem(obj, constr)
        # prob.solve(solver = cp.GUROBI, verbose=False, solver_opts=solver_options)
        # prob.solve(solver=cp.GUROBI)
        prob.solve(solver=OSQP)
        return delta_c.value

n_times = 10

a5 = time.time()
A = data["train"]["A"]
x_dim, y_dim = A.shape[1], A.shape[0]
lamb = cp.Variable(y_dim)
mu = cp.Variable(x_dim)
delta_c = cp.Variable(x_dim)
x_sol = cp.Parameter(x_dim)

c = cp.Parameter(x_dim)
constr = [A.T@(lamb) -delta_c-c + mu == 0.0, cp.multiply(x_sol, mu)==0.0,  mu+x_sol>=1.0]
# constr = [A.T@(lamb) -delta_c-c+ mu == 0.0, mu >= 0, mu*x_sol==0.0, mu>=1.0]
obj = cp.Minimize(cp.sum((delta_c)**2))
prob = cp.Problem(obj, constr)

for i in range(n_times):
    print(i)
    for index in range(200):
        x_sol.value = data["train"]["sol"][index]
        c.value = data["train"]["c"][index] + np.random.normal(0.0, 1.0, c.shape)
        prob.solve(solver=OSQP, warm_start=True)

a6 = time.time()-a5

a7 = time.time()
# for i in range(n_times):
    # print(i)
    # c = data["train"]["c"]
    # c += np.random.normal(0.0, 1.0, c.shape)
    # for index in range(200):
    #     define_lp_and_solve_KKT(c[index], data["train"]["A"], data["train"]["sol"][index])

a8 = time.time()-a7

print("Time spent on QP: {}, QP(random c): {}".format(a6, a8))