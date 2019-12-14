# Annotated example of usage of this package.
from core import *
from visualise import *
import matplotlib.pyplot as plt

# 1. Train a VGG16 network on CIFAR 100. Let's train for 200 epochs (this will take a while - on test computer with
# NVidia GeForce RTX 2080 Ti, each epoch of training takes ~ 10 seconds))
train_network(
    dir='result/VGG16-CIFAR100/',
    dataset='CIFAR100',
    data_path='data/',
    epochs=200,
    model='VGG16',
    optimizer='SGD',
    optimizer_kwargs={
        'lr': 0.1,
        'momentum': 0.1,
        'weight_decay': 5e-4
    }
)

# 2. After this step, you should have a bunch of stats- and checkpoint files under the chosen dir. In this case, they
# are stored under .result/VGG16-CIFAR100. The stats files contains the key information of the training and testing (if
# that epoch is scheduled for testing) information, where the checkpoint-00XXX.pt contains the state_dict of the model
# and the optimizer that we need for later analyses.

# 3. Let's consider the spectrum on the 200th epoch (last training epoch)

# Let's first use the Lanczos estimation
lanc = compute_eigenspectrum(
    dataset='CIFAR100',
    data_path='data/',
    model='VGG16',
    checkpoint_path='result/VGG16-CIFAR100/checkpoint-00200.pt',
    save_spectrum_path='result/VGG16-CIFAR100/spectra/spectrum-00200-ggn_lanczos',
    save_eigvec=True,
    curvature_matrix='ggn_lanczos',
)

# We compare it against the Monte Carlo sampling of diagonal approximation
diag = compute_eigenspectrum(
    dataset='CIFAR100',
    data_path='data/',
    model='VGG16',
    checkpoint_path='result/VGG16-CIFAR100/checkpoint-00200.pt',
    save_spectrum_path='result/VGG16-CIFAR100/spectra/spectrum-00200-ggn_diag_mc',
    save_eigvec=True,
    curvature_matrix='ggn_diag_mc',
)

# 4. Compare the difference by plotting the two approximations
plot_spectrum('diag', diag)
plot_spectrum('lanczos', lanc)
plt.show()
