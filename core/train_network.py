import torch
from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG, SWA
import optimizers
import numpy as np
import os, time
import tabulate


def train_network(
        dir,
        dataset,
        data_path,
        model: str,
        optimizer: str = 'SGD',
        optimizer_kwargs: dict = None,
        use_test: bool = True,
        batch_size: int = 128,
        num_workers: int = 4,
        resume: str = None,
        epochs: int = 300,
        save_freq: int = 25,
        eval_freq: int = 5,
        schedule: str = 'linear',
        swag: bool = False,
        swag_no_cov: bool = True,
        swag_resume: str = None,
        swag_subspace: str = 'pca',
        swag_lr: float = 0.05,
        swag_rank: int = 20,
        swag_start: int = 161,
        swag_c_epochs: int = 1,
        verbose: bool = False,
        device: str = 'cuda',
        seed: int = None
):
    """
    This function trains a neural network model with given model, dataset, optimiser and other relevant configurations.
    Parameters
    ----------
    dir: str: the directory to which the models and statistics are saved

    dataset: str: ['CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet32'*]: the dataset on which you would like to train the
    model. For ImageNet 32, we use the downsampled 32 x 32 Full ImageNet dataset. We do not provide download due to
    the proprietary issues, and please drop the data of ImageNet 32 in 'data/' folder

    data_path: str: the path string of the dataset

    model: str: the neural network architecture you would like to train. All available models are listed under 'models'/
    Example: VGG16BN, PreResNet110 (Preactivated ResNet - 110 layers)

    optimizer: str: the optimizer you would like to use. In additional to all the standard optimizers defined under
    torch.optim.Optimizer, in optimizer/ we defined some additional optimizers that you may use. Currently we only
    included SGD for the torch in-built optimizer. you may import yours manually by specifying the optimizer under
    optimizers/__init__.py

    optimizer_kwargs: dict: the keyword arguments to be supplied to the optimizer object. Some common ones include
    learning rate 'lr', momentum, weight decay, etc that are often optimizer-specific

    use_test: bool: if True, you will test the model on the test set. If not, a portion of the training data will be
    assigned as the validation set.

    batch_size: int: the minibatch size

    num_workers: int: number of workers for the dataloader

    resume: str: If not None, this string specifies a checkpoint containing the state-dict of the optimizer and the
    model from which pytorch may resume training

    epochs: int: total number of epochs of training

    save_freq: int: how frequent to save the model.
    Caution: for highly complicated modern models with many parameters, saving too often may quickly take up storage
    space.

    eval_freq: int: how frequent should the model evaluate on the validation/test dataset

    schedule: learning rate schedule. Allowed command = 'linear': linear decaying learning rate schedule and 'None':
    constant learning rate

    swag: whether to use Stochastic Weight Averaging (Gaussian)

    swag_no_cov: if True, no covariance matrix will be generated and we only have Stochastic Weight Averaging (instead
    of SWA-Gaussian)

    swag_resume: similar to ''resume'' argument, but on the SWA(G) model

    swag_subspace: *only applicable if swag=True and swag_no_cov=False'* subspace of the SWAG model

    swag_lr: *only applicable if swag=True* the learning rate after swa is activated.

    swag_rank:  *only applicable if swag=True and swag_no_cov=False'* rank of SWAG Gaussian approx

    swag_start: *only applicable if swag=True*: the starting epoch number of weight averaging

    swag_c_epochs:  *only applicable if swag=True*: frequency of model collection for averaging

    verbose: if True, verbose and debugging information will be displayed

    device: ['cpu', 'cuda']: the device on which the model and all computations are performed. Strongly recommend 'cuda'
    for GPU accleration in CUDA-enabled Nvidia Devices

    seed: if not None, a manual seed for the pseudo-random number generation will be used.

    Returns
    -------

    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'
    print('Preparing directory %s' % dir)
    os.makedirs(dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    print('Using model ', model)
    model_cfg = getattr(models, model)

    loaders, num_classes = data.loaders(
        dataset,
        data_path,
        batch_size,
        num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not use_test,
    )

    print('Preparing model')
    print(*model_cfg.args, dict(**model_cfg.kwargs))
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model.to(device)

    if swag:
        if not swag_no_cov:
            print('SWA-Gaussian Enabled')
            swag_model = SWAG(model_cfg.base,
                              subspace_type=swag_subspace, subspace_kwargs={'max_rank': swag_rank},
                              *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            swag_model.to(device)
        else:
            print('SWA Enabled')
            swag_model = SWA(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            swag_model.to(device)
    print(optimizer + ' training')

    def scheduler(epoch, mode):
        if mode == 'constant':
            return optimizer_kwargs['lr']
        elif mode == 'linear':
            t = epoch / (swag_start if swag else epochs)
            lr_ratio = swag_lr / optimizer_kwargs['lr'] if swag else 0.01
            if t <= 0.5:
                factor = 1.0
            elif t <= 0.9:
                factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
            else:
                factor = lr_ratio
            return optimizer_kwargs['lr'] * factor
        else:
            raise NotImplementedError

    # Initialise a criterion
    criterion = losses.cross_entropy

    # Initialise the optimizer
    o = getattr(optimizers, optimizer)
    optim = o(
        model.parameters(),
        **optimizer_kwargs
    )

    start_epoch = 0
    if resume is not None:
        print('Resume training from %s' % resume)
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if swag and swag_resume is not None:
        checkpoint = torch.load(swag_resume)
        swag_model.load_state_dict(checkpoint['state_dict'])

    utils.save_checkpoint(
        dir,
        start_epoch,
        epoch=start_epoch,
        state_dict=model.state_dict(),
        optimizer=optim.state_dict()
    )

    for epoch in range(start_epoch, epochs):
        time_ep = time.time()

        lr = scheduler(epoch, schedule)
        utils.adjust_learning_rate(optim, lr)
        train_res = utils.train_epoch(loaders['train'], model, criterion, optim, verbose=verbose)

        # update batch norm parameters before testing
        utils.bn_update(loaders['train'], model)

        if epoch == 0 or epoch % eval_freq == eval_freq - 1 or epoch == epochs - 1:
            test_res = utils.eval(loaders['test'], model, criterion)
        else:
            test_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

        if swag and (epoch + 1) > swag_start and (epoch + 1 - swag_start) % swag_c_epochs == 0:
            swag_model.collect_model(model)
            if epoch == 0 or epoch % eval_freq == eval_freq - 1 or epoch == epochs - 1:
                swag_model.set_swa()
                utils.bn_update(loaders['train'], swag_model)
                swag_res = utils.eval(loaders['test'], swag_model, criterion)
            else:
                swag_res = {'loss': None, 'accuracy': None, "top5_accuracy": None}

        if (epoch + 1) % save_freq == 0:
            utils.save_checkpoint(
                dir,
                epoch + 1,
                epoch=epoch + 1,
                state_dict=model.state_dict(),
                optimizer=optim.state_dict()
            )
            utils.save_weight_norm(
                dir,
                epoch + 1,
                name='weight_norm',
                model=model
            )
            if swag and (epoch + 1) > swag_start:
                utils.save_checkpoint(
                    dir,
                    epoch + 1,
                    name='swag',
                    epoch=epoch + 1,
                    state_dict=swag_model.state_dict(),
                )
                utils.save_weight_norm(
                    dir,
                    epoch + 1,
                    name='swa_weight_norm',
                    model=swag_model
                )

        time_ep = time.time() - time_ep
        memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

        values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
                  test_res['top5_accuracy'], time_ep, memory_usage]

        np.savez(
            dir + 'stats-' + str(epoch),
            train_loss=train_res['loss'],
            time_ep=time_ep,
            memory_usage=memory_usage,
            train_accuracy=train_res['accuracy'],
            train_top5_accuracy=train_res['top5_accuracy'],
            test_loss=test_res['loss'],
            test_accuracy=test_res['accuracy'],
            test_top5_accuracy=test_res['top5_accuracy']
        )

        if swag:
            values = values[:-2] + [swag_res['loss'], swag_res['accuracy'], swag_res['top5_accuracy']] + values[-2:]
            np.savez(
                dir + 'stats-' + str(epoch),
                train_loss=train_res['loss'],
                time_ep=time_ep,
                memory_usage=memory_usage,
                train_accuracy=train_res['accuracy'],
                train_top5_accuracy=train_res['top5_accuracy'],
                test_loss=test_res['loss'],
                test_accuracy=test_res['accuracy'],
                test_top5_accuracy=test_res['top5_accuracy'],
                swag_loss=swag_res['loss'],
                swag_accuracy=swag_res['accuracy'],
                swag_top5_accuracy=swag_res['top5_accuracy']
            )

        if swag:
            values = values[:-2] + [swag_res['loss'], swag_res['accuracy'], swag_res['top5_accuracy']] + values[-2:]
        columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'te_top5_acc', 'time', 'mem_usage']
        if swag:
            columns = columns[:-2] + ['swa_te_loss', 'swa_te_acc', 'swa_te_top5_acc'] + columns[-2:]
            swag_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    if epochs % save_freq != 0:
        utils.save_checkpoint(
            dir,
            epochs,
            epoch=epochs,
            state_dict=model.state_dict(),
            optimizer=optim.state_dict()
        )
        if swag:
            utils.save_checkpoint(
                dir,
                epochs,
                name='swag',
                epoch=epochs,
                state_dict=swag_model.state_dict(),
            )

