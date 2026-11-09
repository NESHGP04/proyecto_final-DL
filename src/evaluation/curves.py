"""Gráficas de curvas de entrenamiento (la usan M1 y M3)"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(history, path, title=""):
    """history: lista de dicts con epoch, train_loss, val_loss, val_macro_f1 (y train_macro_f1 opcional)."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set(xlabel="época", ylabel="pérdida", title="Pérdida")
    ax1.legend()
    if "train_macro_f1" in history[0]:
        ax2.plot(epochs, [h["train_macro_f1"] for h in history], label="train")
    ax2.plot(epochs, [h["val_macro_f1"] for h in history], label="val")
    ax2.set(xlabel="época", ylabel="F1 macro", title="F1 macro", ylim=(0, 1.02))
    ax2.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
