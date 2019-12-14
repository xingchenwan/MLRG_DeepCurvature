import argparse
import tabulate
import time
import numpy as np
import os
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

parser.add_argument('--num_samples', type=int, default=None, metavar='N', help='number of data points to use (default: the whole dataset)')
parser.add_argument('--subsample_seed', type=int, default=None, metavar='N', help='random seed for dataset subsamling (default: None')
parser.add_argument('--stats_batch', type=int, default=1, metavar='B', help='batch size used to compute the statistics (default: 1)')

parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT',
                    default=None,
                    #[
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00000.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00025.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00050.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00075.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00100.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00125.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00150.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00175.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00200.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00225.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00250.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00275.pt",
                        #"/mnt/08B82010B81FFAC0/jade_results/KFAC_Models/PreResNet110_KFAC_ckpts/checkpoint-00300.pt",
                    #],
                    help='checkpoint to load model (default: None)')
parser.add_argument('--swag', action='store_true')

parser.add_argument('--save_path', type=str, default=None, required=True, help='path to npz results file')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--curvature_matrix', type=str, default='gn', help='type of curvature matrix (options: hessian, gn)')
args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
   args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

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
    batch_size=args.stats_batch,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

batch_loader = torch.utils.data.DataLoader(
    datasets['train'],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    datasets['test'],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)


print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
if not args.swag:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model.to(args.device)
    swag_model = None
else:
    swag_model = SWAG(model_cfg.base,
                 subspace_type='random',
                 *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swag_model.to(args.device)
    model = None


criterion = losses.cross_entropy

stat_labels = [
    'train_loss', 'train_acc', 'test_loss', 'test_acc',
    'loss_mean', 'loss_var',
    'grad_mean_norm_sq', 'grad_var',
    'hess_mean_norm_sq', 'hess_var', 'hess_mu',
    'delta', 'alpha'
]

# Is args.ckpt a directory?
if len(args.ckpt) == 1 and os.path.isdir(args.ckpt[0]):
    ckpt = []
    for filename in os.listdir(args.ckpt[0]):
        if filename.endswith(".pt"):
            ckpt.append(os.path.join(args.ckpt[0], filename))
    print("File list: ", ckpt)
else:
    ckpt = args.ckpt

K = len(ckpt)
stat_dict = {
    label: np.zeros(K) for label in stat_labels
}

columns = ['#'] + stat_labels + ['time']

for i, ckpt_path in enumerate(ckpt):
    start_time = time.time()
    print('Loading %s' % ckpt_path)
    checkpoint = torch.load(ckpt_path)
    if not args.swag:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        swag_model.load_state_dict(checkpoint['state_dict'], strict=False)
        swag_model.set_swa()
        model = swag_model.base_model

    utils.bn_update(full_loader, model)
    train_res = utils.eval(full_loader, model, criterion)
    test_res = utils.eval(test_loader, model, criterion)

    stat_dict['train_loss'][i] = train_res['loss']
    stat_dict['train_acc'][i] = train_res['accuracy']
    stat_dict['test_loss'][i] = test_res['loss']
    stat_dict['test_acc'][i] = test_res['accuracy']

    loss_stats = utils.loss_stats(loader, model, criterion, cuda=True, verbose=False,
                                  bn_train_mode=True, curvature_matrix=args.curvature_matrix)

    for label, value in loss_stats.items():
        stat_dict[label][i] = value

    ckpt_time = time.time() - start_time

    values = ['%d/%d' % (i + 1, K)] + [stat_dict[label][i] for label in stat_labels] + [ckpt_time]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='0.2g')
    table = table.split('\n')
    table = '\n'.join([table[1]] + table)
    print(table)

stat_dict['train_err'] = 100.0 - stat_dict['train_acc']
stat_dict['test_err'] = 100.0 - stat_dict['test_acc']

num_parameters = sum([p.numel() for p in model.parameters()])

np.savez(
    args.save_path,
    checkpoints=args.ckpt,
    num_parameters=num_parameters,
    **stat_dict
)
