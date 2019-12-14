import argparse
import time
import tabulate

import numpy as np

from gpytorch.utils.lanczos import lanczos_tridiag

import torch

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str,
                    default="/home/xwan/PycharmProjects/kfac-curvature/out/VGG16/SGD/checkpoint-00300.pt",
                    metavar='CKPT', required=False,
                    help='checkpoint to load model (default: None)')
parser.add_argument('--swag', action='store_true')

parser.add_argument('--iters', type=int, default=2, metavar='N', help='number of lanczos steps (default: 20)')
parser.add_argument('--num_samples', type=int, default=None, metavar='N', help='number of data points to use (default: the whole dataset)')
parser.add_argument('--subsample_seed', type=int, default=None, metavar='N', help='random seed for dataset subsamling (default: None')

parser.add_argument('--bn_train_mode_off', action='store_true')

parser.add_argument('--spectrum_path', type=str,
                    default="/home/xwan/PycharmProjects/kfac-curvature/out/VGG16/SGD/spectrum-00300",
                    metavar='PATH',
                    help='path to save spectrum (default: None)')
parser.add_argument('--basis_path', type=str, default=None, metavar='PATH',
                    help='path to save Lanczos vectors  (default: None)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--curvature_matrix', type=str, default='hessian',
                    help='type of curvature matrix (default: hessian; '
                         'options: '
                         'hessian: Hessian Matrix '
                         'gn (*G*auss *N*ewton)): Generalised Gauss-Newton Matrix '
                         'hessian_diag: Diagonal entries for the Hessian Matrix. This requires the Backpack package '
                         'gn_diag: Diagonal entries for the Gauss-Newton'
                         'gn_diag_mc: Monte Carlo approximation of the Diagonal entires of the Gauss-Newton'
                         'gn_noise, '
                         'hessian_noise.')
parser.add_argument('--weight_only', action='store_true', help='Whether to compute the weights only (i.e. no eigenvalue and eigenvector computations)')

args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
   args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

if args.curvature_matrix in ['gn_diag', 'gn_diag_mc', 'hessian_diag']:
    try:
        import backpack
    except ImportError:
        raise ValueError("To compute hessian_diag or gn_diag, you need to install backpack extension to pytorch.")


torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

datasets, num_classes = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
    train_subset=args.num_samples,
    train_subset_seed=args.subsample_seed,
)

loader = torch.utils.data.DataLoader(
    datasets['train'],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

full_datasets, _ = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
)

full_loader = torch.utils.data.DataLoader(
    full_datasets['train'],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
if not args.swag:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    print('Loading %s' % args.ckpt)
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
else:
    swag_model = SWAG(model_cfg.base,
                 subspace_type='random',
                 *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    print('Loading %s' % args.ckpt)
    checkpoint = torch.load(args.ckpt)
    swag_model.load_state_dict(checkpoint['state_dict'], strict=False)
    swag_model.set_swa()
    model = swag_model.base_model

model.to(args.device)

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
        if self.curvature_matrix == 'hessian':
            output = utils.hess_vec(
                vector,
                self.loader,
                self.model,
                self.criterion,
                cuda=args.device.type=='cuda',
                bn_train_mode=not args.bn_train_mode_off,
            )
        elif self.curvature_matrix == 'gn':
            output = utils.gn_vec(
                vector,
                self.loader,
                self.model,
                self.criterion,
                cuda=args.device.type=='cuda',
                bn_train_mode=not args.bn_train_mode_off
            )
        elif self.curvature_matrix == 'covgrad':
            output = utils.covgrad_vec(
                vector,
                self.loader,
                self.model,
                self.criterion,
                cuda=args.device.type=='cuda',
                bn_train_mode=not args.bn_train_mode_off
            )
        elif self.curvature_matrix == 'hessian_noise':
            if args.num_samples is None or self.full_loader is None:
                raise ValueError("hessian_noise is selected, and a value for num_samples is required (cannot be None)")
            output = utils.hess_noise_vec(
                vector,
                self.full_loader,
                self.loader,
                self.model,
                self.criterion,
                cuda=args.device.type=='cuda',
                bn_train_mode = not args.bn_train_mode_off
            )
        elif self.curvature_matrix == 'gn_noise':
            if args.num_samples is None or self.full_loader is None:
                raise ValueError("gn_noise is selected, and a value for num_samples is required (cannot be None)")
            output = utils.gn_noise_vec(
                vector,
                self.full_loader,
                self.loader,
                self.model,
                self.criterion,
                cuda=args.device.type=='cuda',
                bn_train_mode = not args.bn_train_mode_off
            )
        else:
            raise ValueError("Unrecognised curvature_matrix argument "+self.curvature_matrix)
        time_diff = time.time() - start_time

        self.iters += 1
        print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        # return output.unsqueeze(1)¬
        return output.cpu().unsqueeze(1)

w = torch.cat([param.detach().cpu().view(-1) for param in model.parameters()])
w_l2_norm = torch.norm(w).numpy()
w_linf_norm = torch.norm(w, float('inf')).numpy()


if not args.weight_only:
    # Curvature matrix-vector product
    if args.curvature_matrix not in ['gn_diag', 'hessian_diag', 'gn_diag_mc']:
        if args.curvature_matrix == 'gn' or args.curvature_matrix == 'hessian':
            productor = CurvVecProduct(loader, model, criterion, args.curvature_matrix)
        # Curvature matrix noise-vector product
        elif args.curvature_matrix == 'gn_noise' or args.curvature_matrix == 'hessian_noise':
            productor = CurvVecProduct(loader, model, criterion, args.curvature_matrix, full_loader=full_loader)
        else:
            raise NotImplemented

        utils.bn_update(full_loader, model)
        Q, T = lanczos_tridiag(productor, args.iters, dtype=torch.float32, device='cpu', matrix_shape=(num_parametrs, num_parametrs))

        eigvals, eigvects = T.eig(eigenvectors=True)
        gammas = eigvects[0, :] ** 2
        V = eigvects.t() @ Q.t()

    else:
        from curvature.utils import curv_diag
        from curvature.models.vgg import get_backpacked_VGG
        # Maps the curvature_matrix keyword to the backpack extension name
        kw2ext = {
            'gn_diag': ('DiagGGNExact', ),
            'hessian_diag': ('DiagHessian', ),
            'gn_diag_mc': ('DiagGGNMC', ),
        }
        if 'VGG' in args.model:
            model = get_backpacked_VGG(model, depth=6, num_classes=num_classes)
        result = curv_diag(loader, model, cuda=args.device.type == 'cuda', criterion=criterion,
                           bn_train_mode=not args.bn_train_mode_off, extensions=kw2ext[args.curvature_matrix])
        # Not really eigvals and eigvecs, but just to preserve the continuity of the code
        eigvals = list(result.values())[0]
        gammas = None
        V = None

    if args.spectrum_path is not None:
        np.savez(
            args.spectrum_path,
            eigvals=eigvals.numpy() if eigvals is not None else None,
            gammas=gammas.numpy() if gammas is not None else None
        )

    if args.basis_path is not None:
        torch.save(
            {
                'w': w,
                'eigvals': eigvals if eigvals is not None else None,
                'gammas': gammas if gammas is not None else None,
                'V': V if V is not None else None,
                'w_l2_norm': w_l2_norm,
                "w_inf_norm": w_linf_norm
            },
            args.basis_path,
        )
    order = np.argsort(np.abs(eigvals.numpy()[:, 0]))[::-1]
    table = [[eigvals[order[i], 0].item(), gammas[order[i]].item()] for i in range(order.size)]
    print(tabulate.tabulate(table, headers=['value', 'weight'], tablefmt='simple'))
    #print('max val = '+np.max(eigvals.numpy()[:, 0]))
else:
    if args.basis_path is not None:
        torch.save(
            {
                'w': w,
                'w_l2_norm': w_l2_norm,
                "w_inf_norm": w_linf_norm
            },
            args.basis_path,
        )
    print("L2/L-inf weight norm: ", w_l2_norm, w_linf_norm)

