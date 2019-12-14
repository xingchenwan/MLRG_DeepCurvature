import os
from collections import defaultdict
import itertools
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from optimizers import KFACOptimizer
from backpack import extend, backpack
from backpack.extensions import DiagGGN


def save_checkpoint(dir, index, name='checkpoint', **kwargs):
    filepath = os.path.join(dir, '%s-%05d.pt' % (name, index))
    state = dict(**kwargs)
    torch.save(state, filepath)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_kfac_damping(optimizer, damping):
    for param_group in optimizer.param_groups:
        param_group['damping'] = damping
    return damping


def adjust_learning_rate_and_momentum(optimizer, lr, momentum):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['momentum'] = momentum
    return lr, momentum


def train_epoch(loader, model, criterion, optimizer, cuda=True, verbose=False, subset=None, backpacked_model=False,
                *backpack_extensions):
    """
    Train the model with one pass over the entire dataset (i.e. one epoch)
    :param loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param cuda:
    :param verbose:
    :param subset:
    :param backpacked_model: toggle to true if the model has additional backpack functionality
    :param backpack_extensions: the backpack extensions you would like to enable
    :return:
    """
    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_sum = 0.0
    stats_sum = defaultdict(float)
    correct_1 = 0.0
    correct_5 = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    extensions = []
    if backpacked_model and len(backpack_extensions) != 0:
        for extension in backpack_extensions:
            assert extension in backpack.extensions, str(extension) + " is not found in backpack.extensions list!"
            e = getattr(backpack.extensions, extension)
            extensions.append(e())

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            #input = input.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            #target = target.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output, stats = criterion(model, input, target)

        optimizer.zero_grad()

        if isinstance(optimizer, KFACOptimizer) and optimizer.steps % optimizer.TCov == 0:
            # Compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1), 1).squeeze().cuda()
            loss_sample = loss_criterion(output, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()

        # If the list of backpack extension is non-empty
        if len(extensions):
            with backpack.backpack(*extensions):
                loss.backward()
        # Normal step
        else:
            loss.backward()

        optimizer.step()
        loss_sum += loss.data.item() * input.size(0)
        for key, value in stats.items():
            stats_sum[key] += value * input.size(0)

        #pred = output.data.argmax(1, keepdim=True)
        #correct += pred.eq(target.data.view_as(pred)).sum().item()
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_1 += correct[0].view(-1).float().sum(0)
        correct_5 += correct[:5].view(-1).float().sum(0)

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print('Stage %d/10. Loss: %12.4f. Acc: %6.2f. Top 5 Acc: %6.2f' % (
                verb_stage + 1, loss_sum / num_objects_current,
                correct_1 / num_objects_current * 100.0,
                correct_5 / num_objects_current * 100.0
            ))
            verb_stage += 1
        # print(loss_sum / num_objects_current)
    correct_5 = correct_5.cpu()
    correct_1 = correct_1.cpu()
    return {
        'loss': loss_sum / num_objects_current,
        'accuracy': correct_1 / num_objects_current * 100.0,
        'top5_accuracy': correct_5 / num_objects_current * 100.0,
        'stats': {key: value / num_objects_current for key, value in stats_sum.items()}
    }


def eval(loader, model, criterion, cuda=True, verbose=False):
    loss_sum = 0.0
    correct_1 = 0.0
    correct_5 = 0.0
    stats_sum = defaultdict(float)
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            if criterion.__name__ != 'cross_entropy_func':
                loss, output, stats = criterion(model, input, target)
            else:
                model_fn, loss_fn = criterion(model, input, target)
                output = model_fn()
                loss = loss_fn(output)
                stats = {}
            loss_sum += loss.item() * input.size(0)
            for key, value in stats.items():
                stats_sum[key] += value

            #pred = output.data.argmax(1, keepdim=True)
            #correct += pred.eq(target.data.view_as(pred)).sum().item()

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_1 += correct[0].view(-1).float().sum(0) / num_objects_total * 100.0
            correct_5 += correct[:5].view(-1).float().sum(0) / num_objects_total * 100.0

    correct_1 = correct_1.cpu()
    correct_5 = correct_5.cpu()

    return {
        'loss': loss_sum / num_objects_total,
        'accuracy': correct_1,
        'top5_accuracy': correct_5,
        'stats': {key: value / num_objects_total for key, value in stats_sum.items()}
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += input.size(0)

    return {
        'predictions': np.vstack(predictions),
        'targets': np.concatenate(targets)
    }


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)

        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


def _bn_train_mode(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def hess_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=True):
    param_list = list(model.parameters())
    vector_list = []

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)
        loss *= input.size()[0] / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        dL_dvec = torch.zeros(1)
        if cuda:
            dL_dvec = dL_dvec.cuda()
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()
        #print(param_list[0].grad.size())
    model.eval()
    return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)


# Xingchen Wan code addition - 29 Nov
def curv_diag(loader, model_name, model, criterion, num_classes=100, cuda=True, bn_train_mode=True, extensions=None):
    """Compute the hessian/GGN diagonal element. This function uses backpack package and will fail otherwise.
    The model and criterion need to be EXTENDED by backpack before use!
    Note that currently this model only supports AllCNN and VGG16.
    """
    from functools import partial
    from curvature import models
    from curvature.models.vgg import get_backpacked_VGG
    try:
        import backpack
    except ImportError:
        print('this function call requires backpack. Aborting')
        return

    if model_name not in ['VGG16', 'AllCNN_CIFAR100']:
        raise NotImplementedError(str(model_name) + " is not currently supported.")
    if model_name == 'VGG16':
        model = get_backpacked_VGG(model, depth=16, num_classes=num_classes)

    if extensions == 'ggn_diag': extensions = ('DiagGGNExact, ', )
    elif extensions == 'hessian_diag': extensions = ('DiagHessian', )
    elif extensions == 'ggn_diag_mc': extensions = ('DiagGGNMC', )
    else: raise NotImplementedError

    # dictionary between the name of the method and the name of variable
    method2variable = {
        'DiagGGNMC': 'diag_ggn_mc',
        'DiagGGNExact': 'diag_ggn_exact',
        'DiagHessian': 'diag_h'
    }

    bn_extensions = []
    for e in extensions:
        assert e in list(method2variable.keys()), e + 'should be oe of ' + str(list(method2variable.keys()))
        ext_method = getattr(backpack.extensions, e)
        bn_extensions.append(ext_method())

    # Extract the nn.Sequential(.) representation of the model required by the backpack package
    bp_model = backpack.extend(model, debug=False)
    bp_criterion = partial(criterion, backpacked_model=True)

    # bp_model.eval()
    if bn_train_mode:
        bp_model.apply(_bn_train_mode)

    # Create a dictionary of outputs
    result_list = []
    result = {}

    # Initialise each curvature diagonal with list of zero tensors, each of which
    # has the same shape as the parameters of the layers
    param_list = list(bp_model.parameters())
    for param in param_list:
        result_list.append(torch.zeros_like(param).to(param.device))

    # Assign each curvature diagonal with this zero for all curvature diagonal in extensions
    for i in range(len(extensions)):
        if i == 0:
            result[extensions[i]] = result_list
        else:
            result[extensions[i]] = result_list.copy()

    bp_model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        with backpack.backpack(*bn_extensions):
            loss, _, _ = bp_criterion(model, input, target)
            loss *= input.size()[0] / N
            loss.backward()
        for i in range(len(extensions)):
            j = 0
            for param in param_list:
                v = getattr(param, method2variable[extensions[i]])
                result[extensions[i]][j] += v
                j += 1

    # Finally, apply vec(.) operation on the result to obtain a long list of diagonals
    for k, v in result.items():
        result[k] = torch.cat([v_.view(-1) for v_ in v]).view(-1).cpu()
    return result


def covgrad_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=True):
    param_list = list(model.parameters())
    vector_list = []
    # vector_list2 = []

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    # vector2 = torch.zeros_like(vector)
    # for param in param_list:
    #     vector_list2.append(vector2[offset:offset + param.numel()].detach().view_as(param).to(param.device))
    #     offset += param.numel()

    vector_list2 = torch.zeros_like(vector)

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)
        loss *= input.size()[0] / N

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)

        dL_dvec = torch.zeros(1)
        if cuda:
            dL_dvec = dL_dvec.cuda()
            vector_list2.cuda()
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec *= grad_list
        vector_list2 += dL_dvec
        #dL_dvec.backward()
        #print(param_list[0].grad.size())
    model.eval()
    return vector_list2
    #return torch.cat([vector_list2(-1) for vector in vector_list2]).view(-1)


# Xingchen Wan code addition - 20 Nov 2019
def hess_noise_vec(vector, full_loader, batch_loader, model, criterion, cuda=True, bn_train_mode=True):
    """Compute the matrix-vector product between the Hessian noise matrix"""
    full_hess_vec_prod = hess_vec(vector, full_loader, model, criterion, cuda=cuda, bn_train_mode=bn_train_mode)
    batch_hess_vec_prod = hess_vec(vector, batch_loader, model, criterion, cuda=cuda, bn_train_mode=bn_train_mode)
    return full_hess_vec_prod - batch_hess_vec_prod


# Xingchen Wan code addition - 1 Oct 2019
def gn_vec(vector, loader, model, criterion, cuda=True, bn_train_mode=True):
    param_list = list(model.parameters())
    vector_list = []
    num_parameters = sum(p.numel() for p in param_list)

    offset = 0
    for param in param_list:
        vector_list.append(vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    Gv = torch.zeros(num_parameters, dtype=torch.float32, device="cuda" if cuda else "cpu")

    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, output, _ = criterion(model, input, target)
        loss *= input.size()[0] / N

        Jv = R_op(output, param_list, vector_list)
        grad = torch.autograd.grad(loss, output, create_graph=True)
        HJv = R_op(grad, output, Jv)
        JHJv = torch.autograd.grad(
            output, param_list, grad_outputs=HJv, retain_graph=True)
        Gv += torch.cat([j.detach().view(-1) for j in JHJv])
    # model.eval()
    return Gv
    # return torch.cat([param.grad.view(-1) for param in param_list]).view(-1)


# Xingchen Wan code addition - 20 Nov 2019
def gn_noise_vec(vector, full_loader, batch_loader, model, criterion, cuda=True, bn_train_mode=True):
    """Compute the matrix-vector product between the GN noise matrix"""
    full_gn_vec_prod = gn_vec(vector, full_loader, model, criterion, cuda=cuda, bn_train_mode=bn_train_mode)
    batch_gn_vec_prod = gn_vec(vector, batch_loader, model, criterion, cuda=cuda, bn_train_mode=bn_train_mode)
    return full_gn_vec_prod - batch_gn_vec_prod


def R_op(y, x, v):
    """
    Compute the Jacobian-vector product (dy_i/dx_j)v_j. R-operator using the two backward diff trick
    :return:
    """
    if isinstance(y, tuple):
        ws = [torch.zeros_like(y_i).requires_grad_(True) for y_i in y]
    else:
        ws = torch.zeros_like(y).requires_grad_(True)
    jacobian = torch.autograd.grad(y, x, grad_outputs=ws, create_graph=True)
    Jv = torch.autograd.grad(jacobian, ws, grad_outputs=v, retain_graph=True)
    return tuple([j.detach() for j in Jv])


def _gn_vec(model, loss, output, vec, ):
    """Compute the Gauss-newton vector product
    """
    views = []
    offset = 0
    param_list = list(model.parameters())
    for param in param_list:
        views.append(vec[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    vec_ = list(views)

    Jv = R_op(output, param_list, vec_)

    gradient = torch.autograd.grad(loss, output, create_graph=True)
    HJv = R_op(gradient, output, Jv)
    JHJv = torch.autograd.grad(
        output, param_list, grad_outputs=HJv, retain_graph=True)
    Gv = torch.cat([j.detach().flatten() for j in JHJv])
    return Gv

# Xingchen Wan code addition ends


def shrinkage(loader, model, criterion, cuda=True, batch_loader=None, bn_train_mode=True, verbose=True):
    param_list = list(model.parameters())
    num_parameters = sum(p.numel() for p in param_list)

    z = torch.randn(num_parameters).to(param_list[0].device)
    z /= torch.sqrt(torch.sum(z ** 2))

    H_z = hess_vec(
        z,
        batch_loader if batch_loader is not None else loader,
        model,
        criterion,
        cuda=cuda,
        bn_train_mode=bn_train_mode
    )

    mean_value = torch.sum(z * H_z)

    beta = torch.sum((H_z - z * mean_value) ** 2).cpu()

    z_list = []
    offset = 0
    for param in param_list:
        z_list.append(z[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()

    model.eval()
    if bn_train_mode:
        raise NotImplementedError
        model.apply(_bn_train_mode)

    gamma = torch.zeros(1)

    num_batches = len(loader)
    for input, target in tqdm.tqdm(loader):

        model.zero_grad()
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)

        dL_dvec = torch.zeros(1)
        if cuda:
            dL_dvec = dL_dvec.cuda()
        for v, g in zip(z_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()

        H_z_i = torch.cat([p.grad.view(-1) for p in param_list])
        gamma += (torch.sum((H_z - H_z_i) ** 2)).cpu() / num_batches
    model.eval()
    return 1.0 - beta / torch.max(beta, gamma), mean_value, beta, gamma


# Xingchen Wan code modification:
def loss_stats_old(loader, model, criterion, cuda=True, bn_train_mode=True, verbose=False, curvature_matrix='hessian'):
    param_list = list(model.parameters())
    num_parameters = sum(p.numel() for p in param_list)

    model.eval()
    if bn_train_mode:
        # raise NotImplementedError
        model.apply(_bn_train_mode)

    loss_mean = torch.zeros(1)
    loss_sq_mean = torch.zeros(1)

    grad_mean = torch.zeros(num_parameters)
    grad_norm_sq_mean = torch.zeros(1)

    z = torch.randn(num_parameters)
    z /= torch.sqrt(torch.sum(z ** 2))

    H_z_mean = torch.zeros(num_parameters)
    H_z_norm_sq_mean = torch.zeros(1)

    if cuda:
        grad_mean = grad_mean.cuda()
        z = z.cuda()
        H_z_mean = H_z_mean.cuda()

    num_batches = len(loader)
    if verbose:
        loader = tqdm.tqdm(loader)
    for input, target in loader:
        model.zero_grad()
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        grad_i = torch.cat([g.view(-1) for g in grad_list])

        dL_dz = torch.sum(grad_i * z)
        dL_dz.backward()

        H_z_i = torch.cat([p.grad.detach().view(-1) for p in param_list])
        grad_i = grad_i.detach()

        loss_mean += loss.cpu() / num_batches
        loss_sq_mean += loss.cpu() ** 2 / num_batches

        grad_mean += grad_i / num_batches
        grad_norm_sq_mean += torch.sum(grad_i ** 2).cpu() / num_batches

        H_z_mean += H_z_i / num_batches
        H_z_norm_sq_mean += torch.sum(H_z_i ** 2).cpu() / num_batches

    model.eval()

    loss_var = loss_sq_mean - loss_mean ** 2

    grad_mean_norm_sq = torch.sum(grad_mean ** 2).cpu()
    grad_var = grad_norm_sq_mean - grad_mean_norm_sq

    H_z_mean_norm_sq = torch.sum(H_z_mean ** 2).cpu()
    hess_var = H_z_norm_sq_mean - H_z_mean_norm_sq

    hess_mu = torch.sum(z * H_z_mean).cpu()
    delta = torch.sum((H_z_mean - z * hess_mu.item()) ** 2).cpu()
    alpha = torch.max(torch.tensor(0.0), 1.0 - hess_var / num_batches / delta)

    return {
        'loss_mean': loss_mean,
        'loss_var': loss_var,
        'grad_mean_norm_sq': grad_mean_norm_sq,
        'grad_var': grad_var,
        'hess_mean_norm_sq': H_z_mean_norm_sq,
        'hess_var': hess_var,
        'hess_mu': hess_mu,
        'delta': delta,
        'alpha': alpha
    }


def loss_stats(loader, model, criterion, cuda=True, bn_train_mode=True, verbose=False, curvature_matrix='hessian'):
    """
    Compute and save the loss_stats
    :param loader:
    :param model:
    :param criterion:
    :param cuda:
    :param bn_train_mode:
    :param verbose:
    :param curvature_matrix: select the curvature matrix to be used. Available options:
            'hessian' - Hessian matrix
            'gn' - Gauss-Newton matrix
            Other curvature_matrix argument input will result in a ValueError.
    :return:
    Note: for the sake of compatibility, in the final dictionary returned, regardless of the type of curvature matrix used
    the column names will be hess_*, etc.
    """
    param_list = list(model.parameters())
    num_parameters = sum(p.numel() for p in param_list)
    z = torch.randn(num_parameters)
    z /= torch.sqrt(torch.sum(z ** 2))

    vector_list = []
    offset = 0
    for param in param_list:
        vector_list.append(z[offset:offset + param.numel()].detach().view_as(param).to(param.device))

    model.eval()
    if bn_train_mode:
        model.apply(_bn_train_mode)

    loss_mean = torch.zeros(1)
    loss_sq_mean = torch.zeros(1)

    grad_mean = torch.zeros(num_parameters)
    grad_norm_sq_mean = torch.zeros(1)


    H_z_mean = torch.zeros(num_parameters)
    H_z_norm_sq_mean = torch.zeros(1)

    if cuda:
        grad_mean = grad_mean.cuda()
        z = z.cuda()
        H_z_mean = H_z_mean.cuda()

    num_batches = len(loader)
    if verbose:
        loader = tqdm.tqdm(loader)
    for input, target in loader:
        model.zero_grad()
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, output, _ = criterion(model, input, target)

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        grad_i = torch.cat([g.view(-1) for g in grad_list])

        if curvature_matrix == 'hessian':
            dL_dz = torch.sum(grad_i * z)
            dL_dz.backward()
            H_z_i = torch.cat([p.grad.detach().view(-1) for p in param_list])

        elif curvature_matrix == 'gn':
            Jv = R_op(output, param_list, vector_list)
            grad = torch.autograd.grad(loss, output, create_graph=True)
            HJv = R_op(grad, output, Jv)
            JHJv = torch.autograd.grad(
                output, param_list, grad_outputs=HJv, retain_graph=False)
            H_z_i = torch.cat([j.detach().view(-1) for j in JHJv])

        else:
            raise ValueError('Invalid curvature matrix'+curvature_matrix)
        grad_i = grad_i.detach()
        loss_mean += loss.cpu() / num_batches
        loss_sq_mean += loss.cpu() ** 2 / num_batches

        grad_mean += grad_i / num_batches
        grad_norm_sq_mean += torch.sum(grad_i ** 2).cpu() / num_batches
        H_z_mean += H_z_i / num_batches
        H_z_norm_sq_mean += torch.sum(H_z_i ** 2).cpu() / num_batches
    model.eval()

    loss_var = loss_sq_mean - loss_mean ** 2

    grad_mean_norm_sq = torch.sum(grad_mean ** 2).cpu()
    grad_var = grad_norm_sq_mean - grad_mean_norm_sq

    H_z_mean_norm_sq = torch.sum(H_z_mean ** 2).cpu()
    hess_var = H_z_norm_sq_mean - H_z_mean_norm_sq

    hess_mu = torch.sum(z * H_z_mean).cpu()
    delta = torch.sum((H_z_mean - z * hess_mu.item()) ** 2).cpu()
    alpha = torch.max(torch.tensor(0.0), 1.0 - hess_var / num_batches / delta)

    return {
        'loss_mean': loss_mean,
        'loss_var': loss_var,
        'grad_mean_norm_sq': grad_mean_norm_sq,
        'grad_var': grad_var,
        'hess_mean_norm_sq': H_z_mean_norm_sq,
        'hess_var': hess_var,
        'hess_mu': hess_mu,
        'delta': delta,
        'alpha': alpha
    }


def grad(loader, model, criterion, cuda=True, bn_train_mode=False):
    model.eval()
    if bn_train_mode:
        raise NotImplementedError
        model.apply(_bn_train_mode)

    model.zero_grad()
    N = len(loader.dataset)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)
        loss *= input.size()[0] / N
        loss.backward()

    return torch.cat([param.grad.view(-1) for param in model.parameters()]).view(-1)


def loss_stats_layerwise(loader, model, criterion, cuda=True, bn_train_mode=True, verbose=False):
    param_list = list(model.parameters())
    num_parameters = sum(p.numel() for p in param_list)

    model.eval()
    if bn_train_mode:
        raise NotImplementedError
        model.apply(_bn_train_mode)

    z_list = []
    H_z_mean_list = []
    H_z_mean_norm_sq_list = []

    for param in param_list:
        z = torch.randn(param.size())
        z /= torch.sqrt(torch.sum(z ** 2))
        z = z.to(param.device)
        z_list.append(z)
        H_z_mean_list.append(torch.zeros_like(param))
        H_z_mean_norm_sq_list.append(torch.zeros(1).to(param.device))

    num_batches = len(loader)
    if verbose:
        loader = tqdm.tqdm(loader)
    for input, target in loader:
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        loss, _, _ = criterion(model, input, target)

        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)

        for param, grad, z, H_z_mean, H_z_mean_norm_sq in zip(param_list, grad_list, z_list, H_z_mean_list, H_z_mean_norm_sq_list):
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

            dL_dz = torch.sum(grad * z)
            dL_dz.backward(retain_graph=True)

            H_z = param.grad
            H_z_mean += H_z / num_batches
            H_z_mean_norm_sq += torch.sum(H_z ** 2) / num_batches

    alpha_list = []
    delta_list = []
    hess_mu_list = []
    hess_var_list = []
    for z, H_z_mean, H_z_norm_sq_mean in zip(z_list, H_z_mean_list, H_z_mean_norm_sq_list):
        hess_mu = torch.sum(z * H_z_mean)
        hess_var = (H_z_norm_sq_mean - torch.sum(H_z_mean ** 2))

        delta = torch.sum((H_z_mean - hess_mu * z) ** 2)
        alpha = torch.max(torch.tensor(0.0), (1.0 - hess_var / num_batches / delta).cpu())

        hess_mu_list.append(hess_mu.cpu())
        hess_var_list.append(hess_var.cpu())
        delta_list.append(delta.cpu())
        alpha_list.append(alpha.cpu())

    model.eval()

    return {
        'hess_mean_norm_sq_list': H_z_mean_norm_sq_list,
        'hess_var_list': hess_var_list,
        'hess_mu_list': hess_mu_list,
        'delta_list': delta_list,
        'alpha_list': alpha_list
    }


# XW addition
def save_weight_norm(dir, index, name, model):
    """Save the L2 and L-inf norms of the weights of a model"""
    filepath = os.path.join(dir, '%s-%05d.pt' % (name, index))

    w = torch.cat([param.detach().cpu().view(-1) for param in model.parameters()])
    l2_norm = torch.norm(w).numpy()
    linf_norm = torch.norm(w, float('inf')).numpy()
    np.savez(
        filepath,
        l2_norms=l2_norm,
        linf_norms=linf_norm
    )