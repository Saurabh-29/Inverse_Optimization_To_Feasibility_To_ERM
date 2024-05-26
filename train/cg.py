import numpy as np 
import pandas as pd
import sys
import matplotlib.pyplot as plt
from numpy import linalg as LA
import time

index = int(sys.argv[1])
fig, ax = plt.subplots(index, figsize=(50, 50))

name_y = ["Log-loss", "Log-grad_norm", "Log-loss", "Log-grad_norm"]
name_x = ["Iteration", "Iteration", "Time(sec)", "Time(sec)"]

for i in range(index):
    ax[i].set_xlabel(name_x[i])
    ax[i].set_ylabel(name_y[i])
    ax[i].set_yscale('log')
    ax[i].legend()

X = np.array(pd.read_csv("X.csv"))
y_hat = np.array(pd.read_csv("y.csv"))

# fig, ax = plt.subplots(2,2)


def get_next_lambda(prev_l):
    return (1+ np.sqrt(1+4*prev_l**2))/2

def conjugate_gradient(A, b, w, num_iter, eps=0.0, id=-1):
    """
    This algorithm implements the conjugate gradient algorithm. In this algorithm, we make a 
    conjugate direction vector to find the minimum of the function.
    """
    r = b - np.matmul(A, w)
    # print(b.shape, r.shape)
    # exit(0)
    p = r
    iter_l, loss_l, grad_l, lr_l = [], [], [], []
    for i in range(num_iter):
        alpha = np.matmul(r.T, r)/np.matmul(np.matmul(p.T, A), p)
        w = w + alpha*p
        r_new = r - alpha*np.matmul(A, p)
        beta = np.matmul(r_new.T, r_new)/np.matmul(r.T, r)
        p = r_new + beta*p
        loss = get_loss_from_inputs(X, y_hat, w)
        iter_l.append(i)
        loss_l.append(loss[0][0])
        grad_l.append(np.linalg.norm(r_new))
        lr_l.append(alpha)
        r = r_new
        if np.linalg.norm(p)<eps:
            break
    if id>0:
        ax[0].plot(iter_l, loss_l, label='CG')
        ax[1].plot(iter_l, grad_l, label='CG')
    return w

def get_hessian(X):
    return np.matmul(X.T, X)

def get_loss(y, y_hat, w):
    return 0.5*(np.matmul((y-y_hat).T, (y-y_hat)) + np.sum(w**2)/y.shape[0])

def get_grad(y, y_hat, w):
    return np.matmul(X.T, (y-y_hat)) + w/y.shape[0]

def get_loss_from_inputs(X, y_hat, w):
    y = np.matmul(X, w)
    return 0.5*(np.matmul((y-y_hat).T, (y-y_hat)) + np.sum(w**2)/y.shape[0])

def estimate_L_and_mu(X):
    """
    This function estimates the Lipschitz constant of the Hessian matrix of the loss function.
    """
    eigenvalues = np.linalg.eigvals(get_hessian(X))
    return np.max(eigenvalues) +1.0/X.shape[0], np.min(eigenvalues)+1.0/X.shape[0]


def nestorov_gradient_sc(X, y_hat, w, num_iter, kappa, eta):
    """
     This function implements the nestorov gradient algorithm, the difference between this and polyak 
     momentum is that this algorithm uses the previous velocity vector to calculate the momentum term.
    """
    v = w
    w_old = w
    gamma = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)
    for i in range(num_iter):
        # gamma*=0.9
        y = np.matmul(X, v)
        loss = get_loss(y, y_hat, w)
        grad = get_grad(y, y_hat, v)
        v = w + gamma*(w-w_old)
        w_new = v - eta *grad
        print(i, loss, np.linalg.norm(grad))
        w_old = w
        w = w_new
    return w

def nestorov_gradient(X, y_hat, w, num_iter, eta):
    """
     This function implements the nestorov gradient algorithm, the difference between this and polyak 
     momentum is that this algorithm uses the previous velocity vector to calculate the momentum term.
    """
    v = w
    w_old = w
    prev_lambda = 0
    next_lambda = get_next_lambda(prev_lambda)
    iter_l, loss_l, grad_l = [], [], []
    for i in range(num_iter):
        beta = (prev_lambda-1)/next_lambda
        prev_lambda = next_lambda
        next_lambda = get_next_lambda(next_lambda)

        v = w + beta*(w-w_old)
        y = np.matmul(X, v)
        loss = get_loss(y, y_hat, w)
        grad = get_grad(y, y_hat, v)

        w_new = v - eta *grad
        # print(i, loss, np.linalg.norm(grad), beta)
        w_old = w
        w = w_new
        iter_l.append(i)
        loss_l.append(loss[0][0])
        grad_to_see = get_grad(np.matmul(X, w), y_hat, w)
        grad_l.append(np.linalg.norm(grad_to_see))
    ax[0].plot(iter_l, loss_l, label='NA')
    ax[1].plot(iter_l, grad_l, label='NA')
    return w

def heavy_ball_momentum(X, y_hat, w, num_iter, kappa, HBeta):
    """
     This function implements the heavy ball momentum algorithm, the difference between this 
     and the nestorov gradient is that the momentum term is not calculated using the previous 
     weight vector but the previous velocity vector. 
    """
    v = w
    w_old = w
    gamma = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)
    iter_l, loss_l, grad_l = [], [], []
    for i in range(num_iter):
        v = w + gamma*(w-v)
        y = np.matmul(X, w)
        loss = get_loss(y, y_hat, w)
        grad = get_grad(y, y_hat, w)
        w_new = v - grad/L
        print(i, loss, np.linalg.norm(grad))

        w_old = w
        w = w_new
        iter_l.append(i)
        loss_l.append(loss[0][0])
        grad_l.append(np.linalg.norm(grad))
    ax[0].plot(iter_l, loss_l, label='HB')
    ax[1].plot(iter_l, grad_l, label='HB')
    return w

def get_A_and_b(X, y_hat):
    A = get_hessian(X) + np.identity(X.shape[1])/(X.shape[0])
    b = np.matmul(y_hat.T, X).T
    return A, b

L, mu = estimate_L_and_mu(X)
kappa = L/mu
eta = 1.0/L

A, b = get_A_and_b(X, y_hat)
np.random.seed(0)
##Q4..

def get_armijo_lr(X, y, y_hat, w, lr_max=1.0, b=0.99, c=0.5):
    lr = lr_max
    loss = get_loss(y, y_hat, w)
    grad = get_grad(y, y_hat, w)
    ls_loss = loss-(c*np.matmul(grad.T, grad)[0][0])/L
    calls =0
    while(get_loss_from_inputs(X, y_hat, w-lr*grad)>loss-lr*(c*np.matmul(grad.T, grad)[0][0])):
        lr = lr*b
        calls+=1
    return lr, grad, loss, calls

def train_gd_armijo(X, y_hat, w, iter):
    calls = 0
    prev_lr = 1.0/L
    iter_l, loss_l, grad_l, time_l = [], [], [], []
    t1 = time.time()
    for i in range(iter):
        y = np.matmul(X,w)
        lr, grad, loss, ex_calls = get_armijo_lr(X, y, y_hat, w, lr_max=2.0*prev_lr)
        prev_lr=lr
        w = w - lr*grad
        iter_l.append(i)
        loss_l.append(loss[0][0])   
        grad_l.append(np.linalg.norm(grad))
        calls += ex_calls+1
        time_l.append(time.time()-t1)
        print(i, loss, lr, np.linalg.norm(grad))
        print("Calls for iter {} is {}", i, ex_calls)
    ax[0].plot(iter_l, loss_l, label='GD')
    ax[1].plot(iter_l, grad_l, label='GD')
    ax[2].plot(time_l, loss_l, label='GD')
    ax[3].plot(time_l, grad_l, label='GD')

def get_newton_armijo_lr(X, y_hat, loss, grad, d, w, lr_max=1.0, b=0.99, c=0.5):
    lr = lr_max
    ls_loss = loss-(c*np.matmul(grad.T, grad)[0][0])/L
    calls =0
    print(d.shape, grad.shape, w.shape)
    while(get_loss_from_inputs(X, y_hat, w-lr*d)>loss-c*lr*(np.matmul(d.T, grad)[0][0])):
        lr = lr*b
        calls+=1
    return lr, calls


def train_newton_armijo(X, y_hat, w, iter, eps=1e-3):
    calls = 0
    prev_d = w
    lr = 1.0/L
    iter_l, loss_l, grad_l, time_l = [], [], [], []
    t1=time.time()
    for i in range(iter):
        y = np.matmul(X,w)
        loss = get_loss(y, y_hat, w)
        grad = get_grad(y, y_hat, w)
        d = conjugate_gradient(A, grad, prev_d, num_iter=200, eps=eps)
        lr, ex_calls = get_newton_armijo_lr(X, y_hat, loss, grad, d, w, lr_max=2*lr)
        w = w - lr*d
        iter_l.append(i)
        loss_l.append(loss[0][0])
        grad_l.append(np.linalg.norm(grad))
        calls += ex_calls+1
        time_l.append(time.time()-t1)
        print(i, loss, lr, np.linalg.norm(grad))
        print("Calls for iter {} is {}", i, ex_calls)
    ax[0].plot(iter_l, loss_l, label='ND')
    ax[1].plot(iter_l, grad_l, label='ND')
    ax[2].plot(time_l, loss_l, label='ND')
    ax[3].plot(time_l, grad_l, label='ND')

# w = np.random.rand(X.shape[-1],1)
# train_gd_armijo(X, y_hat, w, 100)
# train_newton_armijo(X, y_hat, w, 100)

#Q3...
# w = np.random.rand(X.shape[-1],1)
# num_iter = 200
# w_new = conjugate_gradient(A, b, w.copy(), num_iter)
# print(kappa, eta)
# exit(0)
# w_new = nestorov_gradient(X, y_hat, w.copy(), num_iter, eta)
# w_new = nestorov_gradient_sc(X, y_hat, w.copy(), num_iter, kappa, eta)

# HBeta = 4/((np.sqrt(L)+np.sqrt(mu))**2)
# w_new = heavy_ball_momentum(X, y_hat, w.copy(), num_iter, kappa, HBeta)

def run_Q3():
    w = np.random.rand(X.shape[-1],1)
    num_iter = 600
    _ = conjugate_gradient(A, b, w.copy(), num_iter, id=-1)
    
    _ = nestorov_gradient(X, y_hat, w.copy(), num_iter, eta)
    # w_new = nestorov_gradient_sc(X, y_hat, w.copy(), num_iter, kappa, eta)

    HBeta = 4/((np.sqrt(L)+np.sqrt(mu))**2)
    _ = heavy_ball_momentum(X, y_hat, w.copy(), num_iter, kappa, HBeta)
    plt.legend()
    plt.show()


def run_Q4():
    w = np.random.rand(X.shape[-1],1)
    num_iter=300
    train_gd_armijo(X, y_hat, w, num_iter)
    train_newton_armijo(X, y_hat, w, num_iter, eps=1e-3)
    plt.legend()
    plt.show()

run_Q4()