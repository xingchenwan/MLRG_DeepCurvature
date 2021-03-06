import time
import tabulate
import numpy as np
from gpytorch.utils.lanczos import lanczos_tridiag
import torch
from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG, SWA


def compute_eigenspectrum(
        dataset: str,
        data_path: str,
        model: str,
        checkpoint_path: str,
        curvature_matrix: str = 'hessian_lanczos',
        use_test: bool = True,
        batch_size: int = 128,
        num_workers: int = 4,
        swag: bool = False,
        lanczos_iters: int = 100,
        num_subsamples: int = None,
        subsample_seed: int = None,
        bn_train_mode: bool = True,
        save_spectrum_path: str = None,
        save_eigvec: bool = False,
        seed: int = None,
        device: str = 'cuda',
):
    """
    This function takes a deep learning model and compute the eigenvalues and eigenvectors (if desired) of the deep
    learning model, either using Lanczos algorithm or using Backpack [1] interface of diagonal approximation of the
    various curvature matrix.
    Parameters
    ----------
    dataset: str: ['CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet32'*]: the dataset on which you would like to train the
    model. For ImageNet 32, we use the downsampled 32 x 32 Full ImageNet dataset. We do not provide download due to
    the proprietary issues, and please drop the data of ImageNet 32 in 'data/' folder    data_path

    data_path: str: the path string of the dataset

    model: str: the neural network architecture you would like to train. All available models are listed under 'models'/
    Example: VGG16BN, PreResNet110 (Preactivated ResNet - 110 layers)

    checkpoint_path: str: the path string to the checkpoints generated by train_network, which contains the state_dict
    of the network and the optimizer.

    curvature_matrix: str: the type of curvature matrix and computation method desired.
    Possible values are:
        hessian_lanczos: Lanczos algorithm of Hessian matrix
        ggn_lanczos: Lanczos algorithm on Generalised Gauss-Newton (GGN)
        cov_grad_lancozs: Lanczos algorithm on Covariance of Gradients

        WARNING: the Backpack package (the diagonal computation interface) we use does not support Residual layers in
        ResNets and derived networks (as of 14 Dec 2019),
        Further, it constrains the model to be a subclass of nn.Sequential. We have
        written modified VGG16 for this purpose, but there is no guarantee that other models will work as-is.

    use_test: bool: if True, you will test the model on the test set. If not, a portion of the training data will be
    assigned as the validation set.

    batch_size: int: the minibatch size

    num_workers: int: number of workers for the dataloader

    swag: whether to use Stochastic Weight Averaging (Gaussian)

    lanczos_iters: *only applicable if the curvature_matrix is set to hessian_lanczos, ggn_lanczos or cov_grad_lanczos*
    Number of iterations for the Lanczos algorithm. This also determines the Ritz value - vector pair generated from
    the Eigenspectrum.

    num_subsamples: int: Number of subsamples to draw randomly from the training dataset. If None, the entire dataset
    will be used.

    subsample_seed: int: the Pseudorandom number seed for subsample draw from above.

    bn_train_mode: bool: Applies only if the network architecture (''model'') used contains batch normalization layers.
    Toggles whether BN layers should be in train or eval mode.

    save_spectrum_path: str: If provided, the Ritz value generated (or the diagonal approximation) will be saved to this
    poth.

    save_eigvec: bool: If True, the implied eigenvectors will also be saved to the same format.
    Note: When this is true, instead of converting the arrays to numpy.ndarray we save directly the torch Tensor. The
    eigenvectors have size P, where P is the number of parameters in the model, so turning this mode on while running
    a large number of experiments could take lots of storage.

    seed: if not None, a manual seed for the pseudo-random number generation will be used.

    device: ['cpu', 'cuda']: the device on which the model and all computations are performed. Strongly recommend 'cuda'
    for GPU accleration in CUDA-enabled Nvidia Devices

    Returns
    -------
    (eigvals, gammas, V):
        eigvals: the computed Ritz Value / diagonal elements of the curvature matrix
        gammas:
        V:
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'
    assert curvature_matrix in ['hessian_lanczos', 'ggn_lanczos', 'cov_grad_lanczos',]

    torch.backends.cudnn.benchmark = True
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    print('Using model %s' % model)
    model_cfg = getattr(models, model)

    datasets, num_classes = data.datasets(
        dataset,
        data_path,
        transform_train=model_cfg.transform_test,
        transform_test=model_cfg.transform_test,
        use_validation=not use_test,
        train_subset=num_subsamples,
        train_subset_seed=subsample_seed,
    )

    loader = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    full_datasets, _ = data.datasets(
        dataset,
        data_path,
        transform_train=model_cfg.transform_train,
        transform_test=model_cfg.transform_test,
        use_validation=not use_test,
    )

    full_loader = torch.utils.data.DataLoader(
        full_datasets['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print('Preparing model')
    print(*model_cfg.args, dict(**model_cfg.kwargs))

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

    num_parametrs = sum([p.numel() for p in model.parameters()])

    criterion = losses.cross_entropy

    class CurvVecProduct(object):
        def __init__(self, loader, model, criterion, curvature_matrix, full_loader=None):
            self.loader = loader
            self.full_loader = full_loader
            self.model = model
            self.criterion = criterion
            self.iters = 0
            self.timestamp = time.time()
            self.curvature_matrix = curvature_matrix

        def __call__(self, vector):
            start_time = time.time()
            if self.curvature_matrix == 'hessian_lanczos':
                output = utils.hess_vec(
                    vector,
                    self.loader,
                    self.model,
                    self.criterion,
                    cuda= device == 'cuda',
                    bn_train_mode=bn_train_mode,
                )
            elif self.curvature_matrix == 'ggn_lanczos':
                output = utils.gn_vec(
                    vector,
                    self.loader,
                    self.model,
                    self.criterion,
                    cuda=device == 'cuda',
                    bn_train_mode=bn_train_mode
                )
            elif self.curvature_matrix == 'cov_grad_lanczos':
                output = utils.covgrad_vec(
                    vector,
                    self.loader,
                    self.model,
                    self.criterion,
                    cuda=device == 'cuda',
                    bn_train_mode=bn_train_mode
                )
            else:
                raise ValueError("Unrecognised curvature_matrix argument " + self.curvature_matrix)
            time_diff = time.time() - start_time
            self.iters += 1
            print('Iter %d. Time: %.2f' % (self.iters, time_diff))
            # return output.unsqueeze(1)¬
            return output.cpu().unsqueeze(1)

    w = torch.cat([param.detach().cpu().view(-1) for param in model.parameters()])
    productor = CurvVecProduct(loader, model, criterion, curvature_matrix)
    utils.bn_update(full_loader, model)
    Q, T = lanczos_tridiag(productor, lanczos_iters, dtype=torch.float32, device='cpu',
                           matrix_shape=(num_parametrs, num_parametrs))
    eigvals, eigvects = T.eig(eigenvectors=True)
    gammas = eigvects[0, :] ** 2
    V = eigvects.t() @ Q.t()
    if save_spectrum_path is not None:
        if save_eigvec:
            torch.save(
                {
                    'w': w,
                    'eigvals': eigvals if eigvals is not None else None,
                    'gammas': gammas if gammas is not None else None,
                    'V': V if V is not None else None,
                },
                save_spectrum_path,
            )
        np.savez(
            save_spectrum_path,
            w=w.numpy(),
            eigvals=eigvals.numpy() if eigvals is not None else None,
            gammas=gammas.numpy() if gammas is not None else None
        )
    return {
        'w': w,
        'eigvals': eigvals,
        'gammas' :gammas,
        'V': V
    }