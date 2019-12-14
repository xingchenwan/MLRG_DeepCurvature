import torch
from torch.optim.optimizer import Optimizer, required
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
import time

class LanczosSGD(Optimizer):
    """
        based on sgd optimizer.
    """

    def __init__(self, params, hess_vec_closure, lanczos_steps=5, beta=100, lr=required, dampening=0,
                 weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.hess_vec_closure = hess_vec_closure
        self.lanczos_steps = lanczos_steps
        self.beta = beta

        defaults = dict(lr=lr, dampening=dampening, weight_decay=weight_decay)
        super(LanczosSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #start = time.time()
        loss = None
        if closure is not None:
            loss = closure()

        num_parametrs = sum([p.numel() for group in self.param_groups for p in group['params']])

        grad_list = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is not None:
                    grad_wd = p.grad.data + weight_decay * p.data
                    # grad_list.append(grad_wd.cpu().view(-1))
                    grad_list.append(grad_wd.view(-1))
        grad = torch.cat(grad_list)
        #chk1 = time.time()
        #print("Weight Decay computation ", chk1-start)

        Q, T = lanczos_tridiag(
            self.hess_vec_closure,
            self.lanczos_steps,
            dtype=torch.float32,
            device='cuda',
            matrix_shape=(num_parametrs, num_parametrs)
        )
        #chk2 = time.time()
        #print("Lanczos tridiagonalisation ", chk2 - chk1)

        # eigvals, T_eigvects = lanczos_tridiag_to_diag(T)
        eigvals, T_eigvects = T.eig(eigenvectors=True)
        eigvals = eigvals[:, 0]
        #print(eigvals)
        eigvects = T_eigvects.t() @ Q.t()
        chk3 = time.time()
        #print("Lanczos diagonalisation ", chk3 - chk2)

        overlaps = torch.mm(eigvects, grad.view(-1, 1))
        lambdas = torch.abs(eigvals.view(-1, 1))
        weights = 1.0 / (lambdas + self.beta) - 1.0 / self.beta
        overlaps.mul_(weights)

        final = torch.mm(eigvects.t(), overlaps).view(-1) + grad / self.beta

        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = final[offset:offset+p.numel()].view_as(p).to(p.device)

                p.data.add_(-group['lr'], d_p)
                offset += p.numel()
        #stop = time.time()
        #print("final", stop-chk3)
        return loss
