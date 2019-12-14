import argparse
import torch
import numpy as np
import tabulate
import time

from curvature import data, models, utils, losses
from curvature.methods.swag import SWAG

parser = argparse.ArgumentParser(description='Loss landscape 1d test')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')

parser.add_argument('--basis_path', type=str, default=None, required=True)
parser.add_argument('--save_path', type=str, default=None, required=True, help='path to npz results file')

parser.add_argument('--dist', type=float, default=1.0, metavar='S', help='distance to travel along all directions (default: 60.0)')
parser.add_argument('--N', type=int, default=21, metavar='N', help='number of points on a grid (default: 21)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
    shuffle_train=False,
)

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

num_parameters = sum([p.numel() for p in model.parameters()])

print('Loading %s' % args.basis_path)
basis_dict = torch.load(args.basis_path)

mean = basis_dict['w'].detach().numpy()
eigvals = basis_dict['eigvals'].numpy()[:, 0]
gammas = basis_dict['gammas'].numpy()
V = basis_dict['V'].numpy()
print(V.shape)
rank = eigvals.size

criterion = losses.cross_entropy

idx = np.array([], dtype=np.int32)

idx = np.concatenate((idx, np.argsort(eigvals)[np.minimum(rank - 1, [0, 1, 2, 5])]))
idx = np.concatenate((idx, np.argsort(-eigvals)[np.minimum(rank - 1, [0, 1, 2, 5])]))
idx = np.concatenate((idx, np.argsort(np.abs(eigvals))[np.minimum(rank - 1, [0, 1, 2, 5])]))
idx = np.sort(np.unique(np.minimum(idx, rank - 1)))
print(idx)
K = len(idx)

ts = np.linspace(-args.dist, args.dist, args.N)

train_acc = np.zeros((K, args.N))
train_loss = np.zeros((K, args.N))
test_acc = np.zeros((K, args.N))
test_loss = np.zeros((K, args.N))

columns = ['#', 't', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']

for i, id in enumerate(idx):
    v = V[id, :].copy()
    print(eigvals[id], 1.0 / np.sqrt(np.abs(eigvals[id])), gammas[id])
    print(np.linalg.norm(v))

    for j, t in enumerate(ts):
        start_time = time.time()
        w = mean + t * v

        offset = 0
        for param in model.parameters():
            size = np.prod(param.size())
            param.data.copy_(param.new_tensor(w[offset:offset+size].reshape(param.size())))
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
        print(table)

np.savez(
    args.save_path,
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

