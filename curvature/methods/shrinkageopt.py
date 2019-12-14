import torch
from torch.optim.optimizer import Optimizer, required


class ShrinkageOpt(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 alpha=1.0, mu=0.0, clip_alpha=0.01, origin=None, wd_mode=True, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        alpha=alpha, mu=mu, nesterov=nesterov, origin=origin)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ShrinkageOpt, self).__init__(params, defaults)
        self.wd_mode = wd_mode
        self.clip_alpha = clip_alpha

    def __setstate__(self, state):
        super(ShrinkageOpt, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = max(group['alpha'], self.clip_alpha)
            mu = group['mu']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            origin = group['origin']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if alpha != 1.0:
                    shift = p.data.clone().detach()
                    if origin is not None:
                        shift = p.data.clone().detach() - origin.data
                    if self.wd_mode:
                        d_p.add_((1.0 - alpha) / alpha * mu, shift)
                    else:
                        d_p.mul_(alpha)
                        d_p.add_((1.0 - alpha) * mu, shift)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
