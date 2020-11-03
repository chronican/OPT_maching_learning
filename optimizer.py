import torch
from torch.optim import Optimizer
from constraints import project_lp, LMO_lp
import numpy as np

#code reference:https://pytorch.org
class SGD(Optimizer):
    """
        implement the SGD optimizer used to update the parameters in deep network
        :param params: value and gradients of weights and bias in deep network
        :param kappa:constraint size
        :param step_size: step_size for Frank wofle
        :param l: p for Lp-norm
        :param project: implement projected sgd if true
        :param FW: implement Frank-wolfe algorithm if true
        :param lr: learning rate for sgd and projected sgd
    """
    def __init__(self, params, kappa=1, l=1, step_size=0, project=False, FW=False, lr=0.01):

        self.kappa = kappa
        self.l = l
        self.project = project
        self.fw = FW
        self.k = 0
        self.num_units = 0
        self.step_size = step_size
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, sparsity=False):
        cnt = 0
        thre = 0.000001
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                gradient = p.grad

                if self.fw: # use FWGD updating gradient

                    if self.k == 0 & sparsity: # calculating number of parameters in first step if sparsity is required
                        self.num_units += p.data.numel()
                    s = LMO_lp(p.grad.data.numpy(), self.kappa, self.l) # s for updating gradient

                    if self.step_size == 0: # use adaptive step size
                        d = s - p.data.numpy()
                        g = - p.grad.data.numpy() * d
                        L = np.sqrt(np.power(p.data.numpy(), 2).sum())
                        gamma = np.clip(g / (L * np.power(d, 2)).sum(), 0, 1)

                    elif self.step_size == 1: # use modified step size
                        gamma = 2 / (np.power(self.k, 1.1) + 3)

                    elif self.step_size == 2: # use default step size
                        gamma = 2 / (self.k + 2)

                    else:
                        raise ValueError("Choose from 0, 1, 2 for adaptive learning rate, modified learning rate and default learning rate!")

                    p.data.add_(torch.Tensor(gamma * s - gamma * p.data.numpy()))

                    if sparsity:  # calculting sparsity of the parameteres
                        cnt += (abs(p.data.numpy()) < thre).sum()
                else:
                    p.data.add_(gradient, alpha=-group['lr']) # SGD updating

                    if self.project:
                        if self.l == 1: # project to L1 ball
                            p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 1))

                        if self.l == 2: # project to L2 ball
                            p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 2))

        if self.fw: # updating step size for FWGD
            self.k += 1

        if sparsity: # return sparsity
            return cnt / self.num_units


class SGD_Momentum(Optimizer):
    """
        implement the momentum SGD optimizer used to update the parameters in deep network
        :param params: value and gradients of weights and bias in deep network
        :param kappa:constraint size
        :param l: p for Lp-norm
        :param project:implement projected sgd_momentum if true
        :param lr:learning rate at first
        :param momentum:the weights of the accumulated moment
        :param weight_decay:L2 regularization operator
    """

    def __init__(self, params, kappa=1, l=1, project=False, lr=0.01, momentum=0, weight_decay=0):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.k = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
        super(SGD_Momentum, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                else:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf

                    p.add_(d_p, alpha=-group['lr'])
                    if self.project:
                        if self.l == 1:
                            p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 1))
                        if self.l == 2:
                            p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 2))


class AdaGrad(Optimizer):
    """
        implement the Adaptive gradient descent optimizer used to update the parameters in deep network
        :param params: value and gradients of weights and bias in deep network
        :param kappa:constraint size
        :param l: p for Lp-norm
        :param project:implement projected AdaGrad if true
        :param lr:learning rate at first
        :param delta:constant which make the update stable
        :param weight_decay:L2 regularization operator
    """

    def __init__(self, params, kappa=1, l=1, project=False, FW=False, lr=0.0001, delta=0.000001,
                 weight_decay=0.005):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.k = 0

        defaults = dict(lr=lr, delta=delta, weight_decay=weight_decay)

        super(AdaGrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        r = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            delta = group['delta']
            lr = group['lr']

            for p in group['params']:
                r.append(torch.zeros_like(p.data))

            if weight_decay != 0:
                p.grad = p.grad.add(p, alpha=weight_decay)
            for rs, p in zip(r, group['params']):
                rs[:] = rs + p.grad * p.grad
                p.data = p.data - lr * p.grad / (torch.sqrt(rs) + delta)

                if self.project:
                    if self.l == 1:
                        p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 1))
                    if self.l == 2:
                        p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 2))


class RMSprop(Optimizer):
    """
        implement the RMSprop  optimizer used to update the parameters in deep network
        :param params: value and gradients of weights and bias in deep network
        :param kappa:constraint size
        :param l: p for Lp-norm
        :param project:implement projected RMSprop if true
        :param lr:learning rate at first
        :param delta:constant which make the update stable
        :param weight_decay:L2 regularization operator
    """

    def __init__(self, params, kappa=1, l=1, project=False, lr=0.0001, delta=0.000001,
                 weight_decay=0.005):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.k = 0

        defaults = dict(lr=lr, delta=delta, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        r = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            delta = group['delta']
            lr = group['lr']

            for p in group['params']:
                r.append(torch.zeros_like(p.data))

            if weight_decay != 0:
                p.grad = p.grad.add(p, alpha=weight_decay)
            for rs, p in zip(r, group['params']):
                rs[:] = 0.9 * rs + 0.1 * p.grad * p.grad
                p.data = p.data - group['lr'] * p.grad / (torch.sqrt(rs) + delta)
                if self.project:
                    if self.l == 1:
                        p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 1))
                    if self.l == 2:
                        p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 2))


class ADAM(Optimizer):
    """
        implement the ADAM optimizer used to update the parameters in deep network
        :param params: value and gradients of weights and bias in deep network
        :param kappa:constraint size
        :param l: p for Lp-norm
        :param project:implement projected ADAM if true
        :param lr:learning rate at first
        :param delta:constant which make the update stable
        :param weight_decay:L2 regularization operator
    """

    def __init__(self, params, kappa=1, l=1, project=False, lr=0.0001, delta=0.000001, weight_decay=0.005):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.k = 0
        self.t = 0
        defaults = dict(lr=lr, delta=delta, weight_decay=weight_decay)
        super(ADAM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
        m = []
        v = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            delta = group['delta']
            lr = group['lr']

            for p in group['params']:
                m.append(torch.zeros_like(p.data))
                v.append(torch.zeros_like(p.data))

            if weight_decay != 0:
                p.grad = p.grad.add(p, alpha=weight_decay)

            for ms, vs, p in zip(m, v, group['params']):
                ms[:] = 0.9 * ms + (1 - 0.9) * p.grad.data
                vs[:] = 0.999 * vs + (1 - 0.999) * p.grad.data ** 2
                m_hat = ms / (1 - 0.9 ** (self.t + 1))
                v_hat = vs / (1 - 0.999 ** (self.t + 1))
                p.data = p.data - group['lr'] * m_hat / (torch.sqrt(v_hat + delta))
                if self.project:
                    if self.l == 1:
                        p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 1))
                    if self.l == 2:
                        p.data = torch.Tensor(project_lp(p.data.numpy(), self.kappa, 2))
            self.t = self.t + 1
