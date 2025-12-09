import os
import json
import matplotlib.pyplot as plt

def plot(json_path):

    # Get directory path
    log_dir = os.path.dirname(json_path)

    # Load json log
    with open(json_path, "r") as f:
        data = json.load(f)

    epochs = data["epochs"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    save_path = log_dir+"/loss_curve.png"

    # Plot
    plt.figure(figsize=(10, 6))

    # Training loss
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)

    # Validation loss
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)

    # Axis labels and title
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training & Validation Loss Curve", fontsize=16)

    # Grid and legend
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    json_path = "outputs/2025-12-09_17-16/training_log.json"
    plot(json_path)