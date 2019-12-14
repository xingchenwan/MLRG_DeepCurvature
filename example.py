# Here we provide an example usage of how to use the MLRG DeepCurvature Package
# ---# Script Format #---

from core import *
from visualise import *
import matplotlib.pyplot as plt

# 1. Train a VGG16 network on CIFAR 100. Let's train for 100 epochs (this will take a while - on test computer with
# NVidia GeForce RTX 2080 Ti, each epoch of training takes ~ 10 seconds))
train_network(
    dir='result/VGG16-CIFAR100/',
    dataset='CIFAR100',
    data_path='data/',
    epochs=100,
    model='VGG16',
    optimizer='SGD',
    optimizer_kwargs={
        'lr': 0.03,
        'momentum': 0.9,
        'weight_decay': 5e-4
    }
)

# 2. After this step, you should have a bunch of stats- and checkpoint files under the chosen dir. In this case, they
# are stored under .result/VGG16-CIFAR100. The stats files contains the key information of the training and testing (if
# that epoch is scheduled for testing) information, where the checkpoint-00XXX.pt contains the state_dict of the model
# and the optimizer that we need for later analyses. Lets first visualise the training process
plot_training(
    dir='result/VGG16-CIFAR100/',
    show_top_5=True
)
plt.show()
# 3. Let's consider the spectrum on the 100th epoch (last training epoch)

# Let's first use the Lanczos estimation on the Generalised Gauss-Newton matrix - as a preliminary example, we run 20
# Lanczos interations

lanc = compute_eigenspectrum(
    dataset='CIFAR100',
    data_path='data/',
    model='VGG16',
    checkpoint_path='result/VGG16-CIFAR100/checkpoint-00100.pt',
    save_spectrum_path='result/VGG16-CIFAR100/spectra/spectrum-00100-ggn_lanczos',
    save_eigvec=True,
    lanczos_iters=20,
    curvature_matrix='ggn_lanczos',
)


# 4. Visualise the result using a stem plot
plot_spectrum('lanczos', path='result/VGG16-CIFAR100/spectra/spectrum-00100-ggn_lanczos.npz')
plt.show()

# 5. Visualise loss landscape
build_loss_landscape(
    dataset='CIFAR100',
    data_path='data/',
    model='VGG16',
    spectrum_path='result/VGG16-CIFAR100/spectra/spectrum-00100-ggn_lanczos',
    checkpoint_path='result/VGG16-CIFAR100/checkpoint-00100.pt',
    save_path='result/VGG16-CIFAR100/losslandscape-00100.npz'
)

plot_loss_landscape('result/VGG16-CIFAR100/losslandscape-00100.npz')
plt.show()