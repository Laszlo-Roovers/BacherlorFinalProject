import h5py
import os
import numpy as np
import torch
import random


def prepare_data(DATABASE_PATH, name, split_ratio):
    """Takes a database in HDF5 format and turns in into a normalized train & test set.
    The result is stored locally in ./data/sets/.

    Parameters
    ----------
    DATABASE_PATH : str
        Path of the HDF5 database file.
    name : str
        Reference name with which the resulting datasets should be stored.
    split_ratio : float
        Determines the train-test split.
    """

    # Read HDF5 database into NumPy arrays
    with h5py.File(DATABASE_PATH, "r") as f:
        w_hat = f["w_hat_n_HF"]
        psi_hat = f["psi_hat_n_HF"]
        w = np.fft.ifft2(w_hat).real
        psi = np.fft.ifft2(psi_hat).real

    # Resize to 256x256 for compatibility
    w = w[:, :-1, :-1]
    psi = psi[:, :-1, :-1]

    # Split the array of field snapshots up into a training set and a test set
    split_index = round(split_ratio * len(w))

    w_train, w_test = w[:split_index, :, :], w[split_index:, :, :]
    psi_train, psi_test = psi[:split_index, :, :], psi[split_index:, :, :]

    # Apply normalization to improve numerical stability
    w_train, w_test, psi_train, psi_test = normalize_fields(
        w_train, w_test, psi_train, psi_test
    )

    # Create directory to store results using provided name
    result_path = f"./data/sets/{name}"
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # Save the fully prepared datasets
    np.save(f"{result_path}/wtrain", w_train)
    np.save(f"{result_path}/wtest", w_test)
    np.save(f"{result_path}/psitrain", psi_train)
    np.save(f"{result_path}/psitest", psi_test)


def normalize_fields(w_train, w_test, psi_train, psi_test):
    """Normalize all the fields in the arrays of field snapshots using training set.
    Only summary statistics of the training set are used because the test set should be
    considered as an unknown, that the model does not have access to during training.

    Parameters
    ----------
    w_train : torch.Tensor
        Train set of unnormalized vorticity.
    w_test : torch.Tensor
        Test set of unnormalized vorticity.
    psi_train : torch.Tensor
        Train set of unnormalized stream function.
    psi_test : torch.Tensor
        Test set of unnormalized stream function.

    Returns
    -------
    w_train :torch.Tensor
        Train set of normalized vorticity.
    w_test : torch.Tensor
        Test set of normalized vorticity.
    psi_train : torch.Tensor
        Train set of normalized stream function.
    psi_test : torch.Tensor
        Test set of normalized stream function.
    """
    # Compute relevant statistics of the training data set
    w_train_mean, w_train_std = np.mean(w_train), np.std(w_train)
    psi_train_mean, psi_train_std = np.mean(psi_train), np.std(psi_train)

    # Apply normalization to both training and test data
    #     but with the training stats only!!
    w_train = (w_train - w_train_mean) / w_train_std
    psi_train = (psi_train - psi_train_mean) / psi_train_std

    w_test = (w_test - w_train_mean) / w_train_std
    psi_test = (psi_test - psi_train_mean) / psi_train_std

    return w_train, w_test, psi_train, psi_test


def get_psi_from_w(w):
    """
    Compute the stream function from the vorticity.

    Parameters
    ----------
    w : torch.Tensor
        Vorticity field.

    Returns
    -------
        psi: Stream function.
    """
    # Grid size
    N_HF = 256

    # Conversion from spatial domain to frequency domain
    w_hat = torch.fft.fft2(w)

    # Kernel defintion (TODO: check if description correct)
    kx = 1j * torch.fft.fftfreq(N_HF).reshape(-1, 1) * N_HF
    ky = 1j * torch.fft.fftfreq(N_HF).reshape(1, -1) * N_HF
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1.0
    k_squared = k_squared.to(w_hat.device)
    # Computation of stream function in frequency domain
    psi_hat = w_hat / k_squared
    psi_hat[0, 0] = 0.0

    # Conversion from frequency domain back to spatial domain
    psi = torch.fft.ifft2(psi_hat).real

    return psi


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
