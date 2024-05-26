import cvxpy as cp 
from cvxpy import OSQP
import numpy as np 
import pickle

class Solver():
    def __init__(self, params, cfg=None):
        super(Solver, self).__init__()
        # print(params.keys())
        self.A = params['A']
        self.b = params['b']
        self.margin = params['margin'] if 'margin' in params else 1.0
        self.gamma = params['gamma'] if 'gamma' in params else 0.0
        self.max_cap_on_pred = params['max_cap_on_pred'] if 'max_cap_on_pred' in params else None
        self.task = 'sp' if 'task' not in params else params['task']
        self.cfg = cfg

        if params['problem']=='LP':
            self.problem = self.fwd_solver(self.A, self.b, gamma=self.gamma, task=self.task,
                             max_cap_on_pred=self.max_cap_on_pred)
        elif params['problem']=='QP_KKT':
            self.problem = self.bwd_solver(self.A, self.b, self.margin, self.task)
        

    def fwd_solver(self, A, b, gamma=0.0, task="sp", max_cap_on_pred=None):
        (dim_constraints, dim_target) = A.shape
        self.x = cp.Variable(dim_target)
        if max_cap_on_pred is not None:
            constr = [A@self.x == b, self.x >=0, self.x <= max_cap_on_pred]
        else:
            constr = [A@self.x == b, self.x >=0]
        self.c = cp.Parameter(dim_target)

        if task=='sp':
            if self.gamma==0:
                obj = cp.Minimize(self.c.T@self.x)
            else:
                obj = cp.Minimize(self.c.T@self.x + self.gamma*cp.sum_squares(self.x))
        elif task=='portfolio':
            self.Q = cp.Parameter((dim_target, dim_target), PSD=True)
            obj = cp.Minimize(-self.c.T@self.x + self.gamma*cp.quad_form(self.x, self.Q))
        else:
            print("Not Implemented")
            raise NotImplementedError
           
        problem = cp.Problem(obj, constr)
        return problem
    
    def bwd_solver(self, A, b=None, margin=1.0, task='sp'):
        (dim_constraints, dim_target) = A.shape

        lamb = cp.Variable(dim_constraints)
        mu = cp.Variable(dim_target)
        self.delta_c = cp.Variable(dim_target)
        self.x_binary = cp.Parameter(dim_target)
        self.c = cp.Parameter(dim_target)
        

        if self.task=='sp':
            constr = [A.T@(lamb)-self.delta_c-self.c + mu == 0.0, cp.multiply(self.x_binary, mu)==0.0, mu+self.x_binary>=margin]
            obj = cp.Minimize(cp.sum((self.delta_c)**2))
        elif self.task=='portfolio':
            if self.cfg.q_pred:
                self.x_sol = cp.Parameter(dim_target)
                self.Q = cp.Parameter((dim_target, dim_target))
                self.delta_Q = cp.Variable((dim_target, dim_target))
                constr = [A.T@(lamb)+self.delta_c+self.c + mu == 2*self.gamma*(self.delta_Q+self.Q)@self.x_sol, cp.multiply(self.x_binary, mu)==0.0, mu>=0, mu+self.x_binary>=margin]
                obj = cp.Minimize(cp.sum((self.delta_c)**2) + cp.sum((self.delta_Q)**2))
            else:
                self.x_sol = cp.Parameter(dim_target)
                self.Q = cp.Parameter((dim_target, dim_target))
                constr = [A.T@(lamb)+self.delta_c+self.c + mu == 2*self.gamma*(self.Q@self.x_sol), cp.multiply(self.x_binary, mu)==0.0, mu>=0, mu+self.x_binary>=margin]
                obj = cp.Minimize(cp.sum((self.delta_c)**2))
        else:
            print("Not Implemented")
            raise NotImplementedError

       
        problem = cp.Problem(obj, constr)
        return problem
    
    def binary_round(self, x):
        x = x>1e-6
        x = x.astype(np.float32)*self.margin
        return x

    def solve_QP_KKT(self, c, x_sol, Q=None):
        self.c.value = c
        self.x_binary.value =self.binary_round(x_sol)

        if self.task=='portfolio':
            self.x_sol.value = x_sol
            if Q is None:
                print("Q is None")
                raise NotImplementedError
            self.Q.value = Q
        try:
            # self.problem.solve(solver=OSQP, warm_start=True)
            self.problem.solve(solver=cp.ECOS, warm_start=True)
            # self.problem.solve(solver=cp.GUROBI, warm_start=True)
        except:
            print("Error in solving QP, retrying with GUROBI")
            self.problem.solve(solver=cp.GUROBI, warm_start=True)
        
        if self.task=='portfolio':
            if self.cfg.q_pred:
                return [self.delta_c.value, self.delta_Q.value]
            else:
                return [self.delta_c.value, None]
        else:
            return self.delta_c.value
    
    def solve_LP(self, c, Q=None, return_dual=False):
        self.c.value = c
        if self.task=='portfolio':
            if Q is None:
                print("Q is None")
                raise NotImplementedError
            
            self.Q.value = Q

        try:
            self.problem.solve(warm_start=True)
        except:
            print("Error in solving QP, retrying with GUROBI")
            self.problem.solve(solver=cp.GUROBI, warm_start=True)
        
        # self.problem.solve(solver=cp.GUROBI, warm_start=True)
        if return_dual:
            return self.x.value, self.problem.constraints[1].dual_value
        return self.x.value


def test_solver():
        data_path="../dataset/warcraft_shortest_path_oneskin/12x12/12x12_mini_200.pkl"

        #read data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        print(data.keys())

        params = {"A": data["train"]["A"], "b":data["train"]["b"], "margin":1.0, "problem":"LP"}
        solver = Solver(params)

        import time
        print("begins")
        a1 = time.time()
        for _ in range(5):
            for index in range(200):
                x_sol = solver.solve_LP(data["train"]["c"][index])
                # x_real = data["train"]["sol"][index]
                # print("Diff", (x_sol-x_real).sum())

        print("Time spent on LP: {}".format(time.time()-a1))

        qp_params = {"A": data["train"]["A"], "b":data["train"]["b"], "margin":1.0, "problem":"QP_KKT"}
        solver_qp = Solver(qp_params)
        a2 = time.time()
        for _ in range(5):
            for index in range(200):
                c_delta = solver_qp.solve_QP_KKT(data["train"]["c"][index], data["train"]["sol"][index])

        print("Time spent on QP: {}".format(time.time()-a2))    