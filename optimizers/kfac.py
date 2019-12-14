# Edited by Xingchen Wan: added KFAC-w (decoupled weight decay), KFAC-L2 (L2 regularisation) and Adaptive damping etc.

import math

import torch
import torch.optim as optim
import numpy as np

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True,
                 decoupled_wd=False,
                 adaptive_mode=False,
                 Tadapt=5,
                 omega=19./20.,
                 cuda=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay,)
        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

        self.acc_stats = True
        self.device = 'cuda' if cuda else "cpu"

        # Whether we toggle decoupled weight decay - this is the result of the paper "DECOUPLED WEIGHT DECAY REGULARIZATION"
        # which showed that in Adam, decoupled weight decay demonstrated better result and established that L2 regu-
        # larisation is not equivalent to weight decay for optimisers with a non-identity preconditioning matrix to the
        # gradient update - which is true for all second order method including K-FAC. use this option to activate
        # decoupled style weight decay.
        self.decoupled_weight_decay = decoupled_wd
        self.wd = weight_decay

        # Auto-damping facility: adaptively compute lambda value for damping - requires one additional forward pass
        self.adaptive_mode = adaptive_mode
        # Turning on adaptive mode will activate both adaptive scaling and adaptive damping
        self.omega = omega
        self.Tadapt = Tadapt

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        count = 0
        #print(self.model)
        #print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                #print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.symeig(
            self.m_aa[m], eigenvectors=True)
        self.d_g[m], self.Q_g[m] = torch.symeig(
            self.m_gg[m], eigenvectors=True)

        # XW: squaring the eigenvalues?
        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    def _get_matrix_form_grad(self, m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        return 1) the matrix form of the gradient
        2) the list form of the gradient
        """
        # Xingchen edit on 28 Oct - if using l2 regularisation, the weight norm should be added to the grad before
        # the conditioning step.
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1) + self.wd * m.weight.data.view(m.weight.grad.data.size(0), -1)
            # n_filters * (in_c * kw * kh)
            p_grad_list = [m.weight.grad.detach().to(self.device)]
            #param_list = [m.weight.data.detach().requires_grad_(True)]
        else:
            p_grad_mat = m.weight.grad.data #+ self.l2_reg * m.weight.data
            p_grad_list = [m.weight.grad.detach().to(self.device)]
            #param_list = [m.weight.data.detach().requires_grad_(True)]
        if m.bias is not None:
            bias_grad = m.bias.grad.data.view(-1, 1)
            bias_grad += m.bias.data.view(-1, 1) * self.wd
            p_grad_list = [m.weight.grad.detach().to(self.device), m.bias.grad.detach().to(self.device)]
            p_grad_mat = torch.cat([p_grad_mat, bias_grad], 1)
            #param_list = [m.weight.data.detach().requires_grad_(True), m.bias.data.detach().requires_grad_(True)]

        return p_grad_mat, p_grad_list

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)

        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v, None

    def _kl_clip_and_update_grad(self, updates, lr, ):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            #v[0] *= scaling
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                #v[1] *= scaling
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure=None):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    if self.decoupled_weight_decay:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])

                d_p = p.grad.data

                # If using normal weight decay...
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    if not self.decoupled_weight_decay:
                        d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                # If using decoupled weight decay...


                p.data.add_(-group['lr'], d_p)

    def step(self, model_fn=None, loss_fn=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        natural_grad = {}
        grad = {}
        param = {}

        if model_fn is not None and loss_fn is not None:
            output = model_fn()
            output_d = output.detach().requires_grad_(True)
            loss = loss_fn(output_d)
        elif self.adaptive_mode:
            raise ValueError("Model_fn and loss_fn need to be supplied for adaptive mode")
        else:
            loss, output, output_d = None, None, None

        for m in self.modules:
            classname = m.__class__.__name__
            #if self.adaptive_mode:
            # Add Tikhonov Damping to the A_i and G_i matrices i.e. the Kronecker blocks
            #    self._tikhonov_step(m)
            if self.steps % self.TInv == 0:
                self._update_inv(m)
            p_grad_mat, p_grad_list = self._get_matrix_form_grad(m, classname)
            # Save the gradient - this is required for auto-damping
            grad[m] = p_grad_list
            #param[m] = param_list
            if not self.adaptive_mode:
                v, _ = self._get_natural_grad(m, p_grad_mat, damping)
            else:
                v, _ = self._get_natural_grad(m, p_grad_mat, damping)

            natural_grad[m] = v
            # natural_grad[.] is the unscaled proposal

        if self.adaptive_mode:
            # if self.steps % self.Tadapt == 0:
                # M is the change of objective function under quadratic model
                lr, M = self._rescale_and_get_quadratic_change(natural_grad, loss, output, output_d, grad)
                lr = min(lr, 1)
        self._kl_clip_and_update_grad(natural_grad, lr,)
        group['lr'] = lr

        self._step()
        if self.adaptive_mode and self.steps % self.Tadapt == 0:
            loss = self.auto_lambda(loss_fn, model_fn, loss, M)
        self.steps += 1
        return (loss, output)

    def auto_lambda(self, loss_fn, model_fn, prev_loss, M):
        """Automatically adjust the value of lambda by comparing the difference between the parabolic approximation
        and the true loss
        """
        loss = loss_fn(model_fn())
        # rho - the reduction ratio in Section 6.5
        # print("M", M, "loss_diff", loss-prev_loss)
        rho = (loss - prev_loss) / M
        factor = self.omega ** self.Tadapt
        if rho > 0.75:
            self.param_groups[0]['damping'] *= factor
        elif rho < 0.25:
            self.param_groups[0]['damping'] /= factor
        print(rho, self.param_groups[0]['lr'], self.param_groups[0]['damping'])
        return loss

    def _tikhonov_step(self, m):
        "Regularise A and G for m-th layer using factored Tikhonov - Section 6.3"
        A_norm = torch.trace(self.m_aa[m]) / (self.m_aa[m].shape[0] + 1)
        G_norm = torch.trace(self.m_gg[m]) / self.m_gg[m].shape[0]

        # Compute pi
        pi = torch.sqrt(A_norm / G_norm).cuda()
        # pi = 1.

        # Get /eta (l2 regularisation) and /lambda (damping coefficient)
        eta = torch.tensor(self.wd).cuda()
        lambd = torch.tensor(self.param_groups[0]['damping']).cuda()
        self.m_aa[m].add_(pi * torch.sqrt(eta + lambd), torch.eye(self.m_aa[m].shape[0], device='cuda'))
        self.m_gg[m].add_(torch.sqrt(eta + lambd) / pi, torch.eye(self.m_gg[m].shape[0], device='cuda'))

    def _rescale_and_get_quadratic_change(self, natural_grad, loss, output, output_d, grads):
        """
        Compute scaling (/alpha) in Section 6.4 to the exact F - here we use Generalised Gauss Newton
        Delta: the unscaled natural gradient.
        Here update argument is the /Delta in Section 6.4

        :param natural_grad: the natural gradient (i.e. gradient preconditioned by inverse Fisher)
        :param loss: the network loss
        :param output: the network output - these two are required for the GGN-vector product computation
        :param grads: the gradient without pre-conditioning - this has been computed previously.
        Return:
            M: the predicted change by the quadratic model, under optimal alpha which is just M = 0.5 \nabla h^T(eta)
        """

        # First compute the numerator $-\nabla h^T \Delta$
        grad_delta_product = 0
        natural_grad_list = []
        #param_list = list(params.values())
        param_list = []
        for m in self.modules:
            v = natural_grad[m]
            grad = grads[m]
            grad_delta_product += (v[0] * grad[0]).sum().item()
            natural_grad_list.append(v[0])
            param_list.append(m.weight)
            if m.bias is not None:
                grad_delta_product += (v[1] * grad[1]).sum().item()
                natural_grad_list.append(v[1])
                param_list.append(m.bias)
        # The update tensor - flattened.
        # natural_grad_list = torch.tensor(natural_grad_list).flatten()
        natural_grad_vec = torch.cat([v.flatten() for v in natural_grad_list])
        delta_F = self.ggn_vector_product(natural_grad_list, param_list, loss, output, output_d)
        delta_F_delta = (delta_F * natural_grad_vec).sum().item()
        delta_F_delta += (self.param_groups[0]['damping'] + self.wd) * torch.norm(natural_grad_vec).item()
        # Compute the new scaling factor /alpha
        alpha = grad_delta_product / delta_F_delta
        #print("alpha", alpha)
        #print("grad_delta_product", grad_delta_product)
        return alpha, 0.5 * alpha * grad_delta_product

    def ggn_vector_product(self, vector_list, param_list, loss, output, output_d):
        """
        Compute the GGN-vector product to compute alpha. Code lifted from CurveBall optimiser
        This actually computes v^TGv, which is different from the usual v^T computation
        """
        from torch.autograd import grad
        (Jz,) = self._fmad(output, param_list, vector_list)  # equivalent but slower

        # compute loss gradient Jl, retaining the graph to allow 2nd-order gradients
        (Jl,) = grad(loss, output_d, create_graph=True)
        Jl_d = Jl.detach()  # detached version, without requiring gradients

        # compute loss Hessian (projected by Jz) using 2nd-order gradients
        (Hl_Jz,) = grad(Jl, output_d, grad_outputs=Jz, retain_graph=True)

        # compute J * (Hl_Jz + Jl) using RMAD (back-propagation).
        # note this is still missing the lambda * z term.
        delta_zs = grad(output, param_list, Hl_Jz + Jl_d, retain_graph=True)
        Gv = torch.cat([j.detach().view(-1) for j in delta_zs])
        return Gv

    @staticmethod
    def _fmad(ys, xs, dxs):
        """Forward-mode automatic differentiation - used to compute the exact Generalised Gauss Newton - lifted from CurveBall"""
        v = torch.zeros_like(ys, requires_grad=True)
        g = torch.autograd.grad(ys, xs, grad_outputs=v, create_graph=True)
        return torch.autograd.grad(g, v, grad_outputs=dxs)

    ### An alternative Implementation -
    # https://discuss.pytorch.org/t/adding-functionality-hessian-and-fisher-information-vector-products/23295

    def FisherVectorProduct(self, vector_list, param_list, loss, output, output_d):
        Jv = self.Rop(output, param_list, vector_list)
        batch, dims = output.size(0), output.size(1)
        if loss.grad_fn.__class__.__name__ == 'NllLossBackward':
            outputsoftmax = torch.nn.functional.softmax(output, dim=1)
            M = torch.zeros(batch, dims, dims).cuda() if outputsoftmax.is_cuda else torch.zeros(batch, dims, dims)
            M.reshape(batch, -1)[:, ::dims + 1] = outputsoftmax
            H = M - torch.einsum('bi,bj->bij', (outputsoftmax, outputsoftmax))
            HJv = [torch.squeeze(H @ torch.unsqueeze(Jv[0], -1)) / batch]
        else:
            HJv = self.HesssianVectorProduct(loss, output, Jv)
        JHJv = self.Lop(output, param_list, HJv)

        return torch.cat([torch.flatten(v) for v in JHJv])

    def HesssianVectorProduct(self, f, x, v):
        df_dx = torch.autograd.grad(f, x, create_graph=True, retain_graph=True)
        Hv = self.Rop(df_dx, x, v)
        return tuple([j.detach() for j in Hv])

    @staticmethod
    def Rop(ys, xs, vs):
        if isinstance(ys, tuple):
            ws = [torch.tensor(torch.zeros_like(y), requires_grad=True) for y in ys]
        else:
            ws = torch.tensor(torch.zeros_like(ys), requires_grad=True)

        gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
        re = torch.autograd.grad(gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=True)
        return tuple([j.detach() for j in re])

    @staticmethod
    def Lop(ys, xs, ws):
        vJ = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
        return tuple([j.detach() for j in vJ])