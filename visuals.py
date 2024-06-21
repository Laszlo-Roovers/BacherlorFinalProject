import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_loss(train_loss) -> plt.Figure:
    fig, axes = plt.subplots()
    fig.suptitle("CNN Performance for Turbulent Flow")

    # Training Loss Plot
    axes.plot(train_loss)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("MSE Loss")
    axes.set_title("Training Loss")

    return fig


def compare_truth_prediction(truth : np.array, pred : np.array) -> plt.Figure:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
    axes[0].set_title(r"Ground truth")
    im0 = axes[0].imshow(truth[:, :])
    plt.colorbar(im0, fraction=0.046, pad=0.04)
    axes[1].set_title(r"Prediction")
    im1 = axes[1].imshow(pred[:, :])
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    fig.tight_layout()
    
    return fig
