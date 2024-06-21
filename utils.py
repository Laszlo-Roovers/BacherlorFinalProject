import os
import numpy as np
import torch
import random


def odd_to_even_2D(f_hat):
    # go from N to N-1 points in fourier space
    assert f_hat.shape[0] % 2 == 1
    assert f_hat.shape[0] == f_hat.shape[1]
    N = f_hat.shape[0]
    neqy = N // 2
    f_hat_even = (np.concatenate((f_hat[: neqy + 1, :], f_hat[neqy + 2 :, :]), axis=0) * (N - 1) / N)
    f_hat_even = (np.concatenate((f_hat_even[:, : neqy + 1], f_hat_even[:, neqy + 2 :]), axis=1) * (N - 1) / N)
    f_hat_even[neqy, :] = 0
    f_hat_even[:, neqy] = 0

    return f_hat_even


def get_stats(x : np.array):

    stats = {
        'mean': np.mean(x),
        'std': np.std(x)
    }

    return stats


def standardize_fields(field, stats, backward=False):

    if backward:
        return (field * stats['std'] + stats['mean'])
    else:
        return (field - stats['mean'])/stats['std']
    


def w_to_psi(w: np.array) -> np.array:
    """
    Compute the stream function from the vorticity.

    Parameters
    ----------
    w : torch.Tensor
        Vorticity field.

    Returns
    -------
        psi: torch.Tensor.
    """
    # Compute Fourier transform of w
    w_hat = np.fft.fft2(w)

    # Compute wave numbers
    Nx, Ny = w.shape
    kx = np.fft.fftfreq(Nx).reshape(-1, 1) * Nx
    ky = np.fft.fftfreq(Ny).reshape(1, -1) * Ny

    # Compute Laplacian in Fourier space (prevent division by zero)
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1.0

    # Compute stream function in Fourier space
    psi_hat = -w_hat / k_squared
    psi_hat[0, 0] = 0.0

    # Conversion from Fourier space to spatial domain
    psi = np.fft.ifft2(psi_hat).real

    return psi


def psi_to_w(psi: np.array) -> np.array:
    """
    Compute the vorticity from the stream function.

    Parameters
    ----------
    psi : torch.Tensor
        Stream function.

    Returns
    -------
        psi: torch.Tensor.
    """
    # Compute Fourier transform of psi
    psi_hat = np.fft.fft2(psi)

    # Compute wave numbers
    Nx, Ny = psi.shape
    kx = np.fft.fftfreq(Nx).reshape(-1, 1) * Nx
    ky = np.fft.fftfreq(Ny).reshape(1, -1) * Ny

    # Compute Laplacian is Fourier space
    k_squared = kx**2 + ky**2

    # Compute vorticity in Fourier space
    w_hat = -k_squared * psi_hat

    # Conversion from Fourier space to spatial domain
    w = np.fft.ifft2(w_hat).real

    return w


def save_model(model, model_path, name):
    """
    Take a model and save its state dictionary to the designated data folder.

    Parameters
    ----------
        model: A PyTorch model.
        model_path: Folder where model state dicts get stored.
        name: Identifier of the model
    """
    # Make model directory and create file destination
    os.makedirs(model_path, exist_ok=True)
    file_path = os.path.join(model_path, name + ".tar")

    # Extract model configuration
    model_state_dict = model.state_dict()

    # Save model configuration
    torch.save(model_state_dict, file_path)


def set_seed(seed):
    """
    Set the seed of all libraries that incoporate (pseudo-) randomness at once so that the same research results can be reproduced.

    Parameters
    ----------
        seed: integer value passed on the individual seed setter functions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
