import matplotlib.pyplot as plt


def visualize_loss(train_loss, test_loss):
    fig, axes = plt.subplots(1, 2, sharey=True)
    fig.suptitle("CNN Performance for Turbulent Flow")

    # Training Loss Plot
    axes[0].plot(train_loss)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")

    # Test Loss Plot
    axes[1].plot(test_loss)
    axes[1].set_xlabel("Batch")
    axes[1].set_title("Test Loss")

    return fig
