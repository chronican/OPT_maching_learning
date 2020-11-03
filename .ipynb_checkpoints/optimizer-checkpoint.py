import torch
from torch.optim import Optimizer
from collections import defaultdict
from constraints import project_l1, LMO_lp
import numpy as np

class SGD(Optimizer):
    def __init__(self, params, kappa=1, l=1, step_size =0, project= False, FW=False, lr= 0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.fw = FW
        self.k = 0
        self.step_size = step_size
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if self.fw:
                    s = LMO_lp(p.grad.data.numpy(), self.kappa, self.l)
                    if self.step_size == 0:
                   # 
                        d = s - p.data.numpy()
                        g = - p.grad.data.numpy() * d
                        #L = 6
                        L = np.sqrt(np.power(p.data.numpy(),2).sum())
                        gamma = np.clip(g/(L*np.power(d,2)).sum(),0,1)
                    if self.step_size ==1 :
                        gamma = 2 / (np.power(self.k, 1.1) + 3)
                    if self.step_size == 2:
                        gamma = 2/(self.k+2)
                    #print(gamma)
                    delta_p = torch.Tensor(gamma * s - gamma * p.data.numpy())
                    p.data.add_(delta_p)
                    #print(np.sqrt(np.power(p.data.numpy(),2).sum()))
                    #print(np.abs(p.data.numpy()).sum())
                else:
                    p.add_(d_p, alpha=-group['lr'])
                    #print(np.max(np.abs(p.data.numpy())))
                    #print(np.abs(p.data.numpy()).sum())

                    if self.project:
                        #if self.l == 1:
                            p.data = torch.Tensor(project_l1(p.data.numpy(), self.kappa))
                                                #print(p.data.shape)
                          #  print(np.abs(p.data.numpy()).sum())
                            #print(np.sqrt(np.power(p.data.numpy(),2).sum()))
        if self.fw:
            # if self.k <600*50:
            self.k+=1




class AdaGrad(Optimizer):
    def __init__(self, params, kappa=1, l=1, project= False, FW=False, lr= 0.0001, delta=0.000001,
                 weight_decay=0.005):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.fw = FW
        self.k =0

        defaults = dict(lr=lr, delta=delta,weight_decay=weight_decay)

        super(AdaGrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        r=[]
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            delta=group['delta']
            lr=group['lr']

            for p in group['params']:
                r.append(torch.zeros_like(p.data))

            if weight_decay != 0:
                p.grad = p.grad.add(p, alpha=weight_decay)
                #if self.fw:

                    #s = LMO_l1(p.grad.data.numpy(), self.kappa)
                    #gamma = 2 / (self.k + 2)
                    #delta_p = torch.Tensor(gamma * s - gamma * p.data.numpy())
                    #p.data.add_(delta_p)
            for i,p in zip(range(len(r)),group['params']):
                r[i]=r[i]+p.grad*p.grad
                p.data = p.data - lr * p.grad /(torch.sqrt(r[i])+delta)

                if self.project:
                    if self.l == 1:
                        p.data = torch.Tensor(project_l1(p.data.numpy(), self.kappa))

        return loss

class RMSprop(Optimizer):
    def __init__(self, params, kappa=1, l=1, project= False, FW=False, lr= 0.0001, delta=0.000001,
                 weight_decay=0.005):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.fw = FW
        self.k =0

        defaults = dict(lr=lr, delta=delta,
                        weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        r=[]
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            delta=group['delta']
            lr = group['lr']

            for p in group['params']:
                r.append(torch.zeros_like(p.data))

            if weight_decay != 0:
                p.grad = p.grad.add(p, alpha=weight_decay)


            flag =0
            for i,p in zip(range(len(r)),group['params']):
                if self.fw:
                    s = LMO_l1(p.grad.data.numpy(), self.kappa)

                    gamma = 2 / (self.k + 2)
                    r[i]=0.9*r[i]+0.1*p.grad*p.grad
                    # delta_p = torch.Tensor(gamma * s - gamma * p.data.numpy()/(torch.sqrt(r[i])+delta))
                    # p.data.add_(delta_p)
                    p.data = p.data - gamma *(torch.Tensor(s)-p.data)

                if self.project:
                    if self.l == 1:
                        p.data = torch.Tensor(project_l1(p.data.numpy(), self.kappa))
        if self.fw:
            self.k+=1
        return loss

class ADAM(Optimizer):
    def __init__(self, params, kappa=1, l=1, project= False, FW=False, lr= 0.0001, delta=0.000001,
                 weight_decay=0.005):
        self.kappa = kappa
        self.l = l
        self.project = project
        self.fw = FW
        self.k =0
        self.t = 0
        defaults = dict(lr=lr, delta=delta,
                        weight_decay=weight_decay)
        super(ADAM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        m=[]
        v=[]
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            delta=group['delta']
            lr=group['lr']

            for p in group['params']:
                m.append(torch.zeros_like(p.data))
                v.append(torch.zeros_like(p.data))

            if weight_decay != 0:
                p.grad = p.grad.add(p, alpha=weight_decay)

            for ms,vs,p in zip(m,v,group['params']):
                ms[:] = 0.9 * ms + (1 - 0.9) * p.grad.data
                vs[:] = 0.999 * vs + (1 - 0.999) * p.grad.data ** 2
                m_hat=ms/(1-0.9**(self.t+1))
                v_hat=vs/(1-0.999**(self.t+1))
                if self.fw:
                    s = LMO_l1(p.grad.data.numpy(), self.kappa)
                    gamma = 2 / (self.k + 2)
                    delta_p = torch.Tensor(gamma * s - gamma * p.data.numpy())
                    p.data.add_(delta_p)
                p.data = p.data - group['lr'] * m_hat/ (torch.sqrt( v_hat+ delta))
                if self.project:
                    if self.l == 1:
                        p.data = torch.Tensor(project_l1(p.data.numpy(), self.kappa))
            self.t=self.t+1

        return loss

