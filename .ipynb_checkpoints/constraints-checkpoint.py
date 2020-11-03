import numpy as np
from until import euclidean_proj_l1ball
import math


# l-p norm
def LMO_lp(grad,kappa,p):
    shape = grad.shape
    grad = grad.reshape(-1)
    if p == 1:
        num = 10
        s = np.zeros(grad.shape)
        coord = np.argpartition(np.abs(grad), -num)[-num:]
        s[coord] = kappa * np.sign(grad[coord])/num

    elif p == 2:
        q = 1/(1-1/p)
        norm = np.sqrt(np.power(np.abs(grad),2).sum())
        s = kappa*np.sign(grad)*np.power(np.abs(grad), p/q)/norm

    elif p == math.inf:
        s = kappa * np.sign(grad)

    return - s.reshape(*shape)

def project_l1(grad, kappa):
    shape = grad.shape
    grad = grad.reshape(-1)
    proj = euclidean_proj_l1ball(grad, kappa)
    return proj.reshape(*shape)
