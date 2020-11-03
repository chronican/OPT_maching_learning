import numpy as np
from until import euclidean_proj_l1ball,euclidean_proj_l2ball
import math


def LMO_lp(gradient,kappa,p):
    """
    Calculating LMO for updating parameters in FWGD
    :param gradient: the gradient of parameters from backward propagation
    :param kappa: the constraint size
    :param p: the constraint shape choosing for L1, L2 and L inf norm
    :return s: the result of LMO for updating parameters in gradient descent
    """
    shape = gradient.shape
    gradient = gradient.reshape(-1)
    # L1 norm
    if p == 1:
        num = 10
        s = np.zeros(gradient.shape)
        c = np.argpartition(np.abs(gradient), -num)[-num:]
        s[c] = kappa * np.sign(gradient[c])/num
    # L2 norm
    elif p == 2:
        q = 1/(1-1/p)
        norm = np.sqrt(np.power(np.abs(gradient),2).sum())
        s = kappa*np.sign(gradient)*np.power(np.abs(gradient), p/q)/norm
    # L inf norm
    elif p == math.inf:
        s = kappa * np.sign(gradient)
    else:
        raise ValueError("Please input 1, 2 or math.inf to get LP norm of LMO!")
    return - s.reshape(*shape)

def project_lp(grad,kappa,l):
    """
        return the reshape data after projection into l1-ball or l2-ball
        :param grad: vector which needed to be projected
        :param kappa:constraint size
        :param l:if l is 1,use l1-ball,if l is 2,use l2-ball
        :return : reshape data after projection
        """
    shape = grad.shape
    grad = grad.reshape(-1)
    if l==1:
        proj = euclidean_proj_l1ball(grad, kappa)
    if l==2:
        proj = euclidean_proj_l2ball(grad, kappa)
    return proj.reshape(*shape)

