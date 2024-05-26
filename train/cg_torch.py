import torch

def get_hessian(X):
    return torch.matmul(X.T, X)

def compute_b(y_hat, X, c_t, eta):
    b = torch.matmul(y_hat.T+(c_t*(1/eta-1)).T, X).T
    return b

def compute_A_and_b(X, y_hat, c_t, eta):
    A = get_hessian(X)/eta
    b = compute_b(y_hat, X, c_t, eta)
    return A, b


def conjugate_gradient(A, b, w, num_iter, eps=1e-4):
    """
    This algorithm implements the conjugate gradient algorithm. In this algorithm, we make a 
    conjugate direction vector to find the minimum of the function.
    """
    r = b - torch.matmul(A, w)
    p = r.clone()

    for i in range(num_iter):
        if i == num_iter-1:
            print("loss at iter {}:".format(i), (-w.T@(b-torch.matmul(A, w))).sum())
        rr = torch.tensordot(r, r)
        pAp = torch.tensordot(p, torch.matmul(A,p))

        # alpha = torch.matmul(r.T, r) / torch.matmul(torch.matmul(p.T, A), p)
        alpha = rr/pAp 

        # print("Shapes", w.shape, alpha.shape, p.shape, A.shape, r.shape, rr, pAp)
        w = w + alpha * p
        r_new = r - alpha * torch.matmul(A, p)

        rr_new = torch.tensordot(r_new, r_new)
        # beta = torch.matmul(r_new.T, r_new) / torch.matmul(r.T, r)
        beta = rr_new / rr
        p = r_new + beta * p
        r = r_new.clone()
        if torch.norm(p) < eps:
            print("Converged at iter {} with norm {}".format(i, torch.norm(p)))
            break
        else:
            pass
            print("norm of p at iter {}".format(i), torch.norm(p), torch.norm(w), alpha, beta)
    return w


