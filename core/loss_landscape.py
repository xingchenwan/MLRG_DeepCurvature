import torch
import numpy as np
import tabulate
import time

from curvature import data, models, utils, losses
from curvature.methods.swag import SWAG


def build_loss_landscape(
        dataset: str,
        data_path: str,
        model: str,
        spectrum_path: str,
        checkpoint_path: str,
        use_test: bool = True,
        batch_size: int = 128,
        num_workers: int = 4,
        save_path: str = None,
        dist: float = 1.,
        n_points: int = 21,
        seed: int = None,
        device: str = 'cuda',
        swag: bool = False,
) -> dict:
    """
    This function loads a checkpoint from the network training, and the spectrum result from Lanczos algorithm, and
    perturbs the weight by a specified amount in each of the eigenvalue directions and then store the resulting train/
    testing loss/accuracy after the perturbation. This tool is only for spectrua computed using the Lanczos algorithm

    :param   dataset: str: ['CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet32'*]: the dataset on which you would like to train the
    model. For ImageNet 32, we use the downsampled 32 x 32 Full ImageNet dataset. We do not provide download due to
    the proprietary issues, and please drop the data of ImageNet 32 in 'data/' folder

    :param data_path: str: the path string of the dataset

    :param model: str: the neural network architecture you would like to train. All available models are listed under 'models'/
    Example: VGG16BN, PreResNet110 (Preactivated ResNet - 110 layers)

    :param spectrum_path: str: the output spectrum from the Lanczos eigenspectrum
    Note: only results using Lanczos algorithm can be used; diagonal approximations are not applicable here

    :param checkpoint_path: str: the checkpoint from network training

    :param use_test: bool: if True, you will test the model on the test set. If not, a portion of the training data will be
    assigned as the validation set.

    :param batch_size: int: the minibatch size

    :param num_workers: number of workers for the dataloader

    :param save_path: if provided, the loss stats dictionary will be saved an additional copy as numpy array in the specified
    path.

    :param dist: float. distance to travel along all directions (default: 60.0)

    :param n_points: number of points on a grid (default: 21)

    :param seed:

    :param device:

    :param swag:

    :return:
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'
    torch.backends.cudnn.benchmark = True
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    print('Using model ', model)
    model_cfg = getattr(models, model)

    print('Loading dataset %s from %s' % (dataset, data_path))
    loaders, num_classes = data.loaders(
        dataset,
        data_path,
        batch_size,
        num_workers,
        transform_train=model_cfg.transform_test,
        transform_test=model_cfg.transform_test,
        use_validation=not use_test,
        shuffle_train=False,
    )
    print('Preparing model')

    if not swag:
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        print('Loading %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        swag_model = SWAG(model_cfg.base,
                          subspace_type='random',
                          *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        print('Loading %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        swag_model.load_state_dict(checkpoint['state_dict'], strict=False)
        swag_model.set_swa()
        model = swag_model.base_model

    model.to(device)
    num_parameters = sum([p.numel() for p in model.parameters()])
    print('Loading %s' % spectrum_path)
    basis_dict = torch.load(spectrum_path)

    mean = basis_dict['w'].detach().numpy()
    eigvals = basis_dict['eigvals'].numpy()[:, 0]
    gammas = basis_dict['gammas'].numpy()
    V = basis_dict['V'].numpy()

    rank = eigvals.size
    criterion = losses.cross_entropy
    idx = np.array([], dtype=np.int32)
    idx = np.concatenate((idx, np.argsort(eigvals)[np.minimum(rank - 1, [0, 1, 2, 5])]))
    idx = np.concatenate((idx, np.argsort(-eigvals)[np.minimum(rank - 1, [0, 1, 2, 5])]))
    idx = np.concatenate((idx, np.argsort(np.abs(eigvals))[np.minimum(rank - 1, [0, 1, 2, 5])]))
    idx = np.sort(np.unique(np.minimum(idx, rank - 1)))
    K = len(idx)

    ts = np.linspace(-dist, dist, n_points)

    train_acc = np.zeros((K, n_points))
    train_loss = np.zeros((K, n_points))
    test_acc = np.zeros((K, n_points))
    test_loss = np.zeros((K, n_points))

    columns = ['#', 't', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
    table = None

    for i, id in enumerate(idx):
        v = V[id, :].copy()
        for j, t in enumerate(ts):
            start_time = time.time()
            w = mean + t * v

            offset = 0
            for param in model.parameters():
                size = np.prod(param.size())
                param.data.copy_(param.new_tensor(w[offset:offset + size].reshape(param.size())))
                offset += size

            utils.bn_update(loaders['train'], model)
            train_res = utils.eval(loaders['train'], model, criterion)
            test_res = utils.eval(loaders['test'], model, criterion)

            train_acc[i, j] = train_res['accuracy']
            train_loss[i, j] = train_res['loss']
            test_acc[i, j] = test_res['accuracy']
            test_loss[i, j] = test_res['loss']

            run_time = time.time() - start_time
            values = [id, t, train_loss[i, j], train_acc[i, j], test_loss[i, j], test_acc[i, j], run_time]
            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
            if j == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print('Iteration: '+str(i * len(ts) + j) + '/' + str(len(ts) * len(idx)))
        print(table)

    if save_path is not None:
        np.savez(
            save_path,
            dim=num_parameters,
            ts=ts,
            eigvals=eigvals,
            gammas=gammas,
            idx=idx,
            train_acc=train_acc,
            train_err=100.0 - train_acc,
            train_loss=train_loss,
            test_acc=test_acc,
            test_err=100.0 - test_acc,
            test_loss=test_loss,
        )

    return {
        'dim': num_parameters,
        'ts': ts,
        'eigvals': eigvals,
        'gammas': gammas,
        'idx': idx,
        'train_acc': train_acc,
        'train_err': 100.0 - train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_err': 100.0 - test_acc,
        'test_loss': test_loss,
    }
