# MLRG Deep Curvature

MLRG Deep Curvature is a PyTorch-based package to analyse and visualise neural network curvature and loss landscape.

## Network training and evaluation 

The package provides a range of pre-built modern popular neural network structures, such as VGG and variants of ResNets, and various optimisation schemes in addition to the ones already present in the PyTorch frameworks, such as K-FAC and SWATS. These facilitates faster training and evaluation of the networks (although it is worth noting that any PyTorch-compatible optimisers or architectures can be easily integrated into its analysis framework).
    
## Eigenspectrum analysis of the curvature matrices

Powered by the Lanczos techniques,  with a single random vector the package uses Pearlmutter matrix-vector product trick for fast computation for inference of the eigenvalues and eigenvectors of the common curvature matrices of the deep neural networks. In addition to the standard Hessian matrix, Generalised Gauss-Newton matrix is also supported.
    
## Advanced Statistics of Networks

In addition to the commonly used statistics to evaluate network training and performance such as the training and testing losses and accuracy, the package supports computations of more advanced statistics, such as squared mean and variance of gradients and Hessians (and GGN), squared norms of Hessian and GGN, L2 and L-inf norms of the network weights and etc. These statistics are useful and relevant for a wide range of purposes such as the designs of second-order optimisers and network architecture.
    
## Visualisations

For all main features above, accompanying visualisation tools are included. In addition, with the eigen-information obtained  visualisations of the loss landscape are also supported by studying the sensitivity of the neural network to perturbations of weights. One key difference is that, instead of random directions as featured in some other packages, this package perturbs the weights in the eigenvector directions explicitly.

For an illustrated example of its use, please see example.ipynb.

References:
1. Gardner, J., Pleiss, G., Weinberger, K.Q., Bindel, D. and Wilson, A.G., 2018. Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. In Advances in Neural Information Processing Systems (pp. 7576-7586).

2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and Desmaison, A., 2019. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (pp. 8024-8035).

3. Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

