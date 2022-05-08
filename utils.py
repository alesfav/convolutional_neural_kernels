import torch


def hypersphere_random_sampler(n_points, input_dim):
    """
    Samples inputs uniformly on a d-dimensional hypersphere.

    Args:
        n_points: number of points to sample
        input_dim: dimension of the inputs
    """

    x = torch.randn(n_points, input_dim)
    x /= torch.norm(x, dim=1, keepdim=True)

    return x


def grf_generator(gram, device):
    """
    Generates a centered Gaussian random field with given covariance.

    Args:
        gram: covariance matrix
        device: device to use (cpu or cuda)
    """

    N = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(len(gram)).to(device), gram
    )
    y = N.sample()

    return y


def kernel_regression(K_trtr, K_tetr, y_tr, y_te, ridge, device):
    """
    Computes the generalisation error of kernel regression.

    Args:
        K_trtr: kernel matrix evaluated on the training points
        K_tetr: mixed kernel matrix evaluated on the test and training points
        y_tr: training labels
        y_te: test labels
        ridge: L2 regularizer
        device: device to use (cpu or gpu)
    """

    alpha = torch.linalg.inv(K_trtr + ridge * torch.eye(y_tr.size(0)).to(device)) @ y_tr
    f = K_tetr @ alpha
    mse = (f - y_te).pow(2).mean()

    return mse


def kernel_spectrum(gram):
    """
    Computes the eigenvalues of a kernel matrix (returned in decreasing order).

    Args:
        gram: kernel matrix to diagonalise
    """

    evals, _ = torch.linalg.eigh(gram)
    evals = 1 / gram.size(-1) * torch.flip(evals, dims=(0,))

    return evals
