import matplotlib.pyplot as plt
from src.constants import FIGS


def plot_model_history(history, label_1: str, label_2: str, model: str):
    """
    The following function is written by referring the following
    plot in the following code
    https://www.kaggle.com/code/shamsheerrahiman/cnn-birds-resnet50
    on April-12-2023
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history[label_1])
    plt.plot(history.history[label_2])
    plt.title(f"Models {label_1}")
    plt.ylabel(f"{label_1}")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"{FIGS}/{model}_{label_1}.png")
