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

    # Zoomed plot: epoch 50 onwards, y-axis clipped to show overfitting gap
    save_path_zoom = log_dir + "/loss_curve_zoom.png"
    zoom_start = 50
    zoom_epochs = [e for e in epochs if e >= zoom_start]
    zoom_train  = [train_loss[i] for i, e in enumerate(epochs) if e >= zoom_start]
    zoom_val    = [val_loss[i]   for i, e in enumerate(epochs) if e >= zoom_start]

    if zoom_epochs:
        # Smooth val loss with moving average
        def smooth(values, window=20):
            result = []
            for i in range(len(values)):
                start = max(0, i - window // 2)
                end   = min(len(values), i + window // 2 + 1)
                result.append(sum(values[start:end]) / (end - start))
            return result

        zoom_val_smooth = smooth(zoom_val, window=20)

        plt.figure(figsize=(10, 6))
        plt.plot(zoom_epochs, zoom_train, label="Train Loss", linewidth=2)
        plt.plot(zoom_epochs, zoom_val, color="orange", alpha=0.2, linewidth=1)
        plt.plot(zoom_epochs, zoom_val_smooth, color="orange", label="Val Loss (smoothed)", linewidth=2)
        # Find best val epoch
        best_idx  = zoom_val.index(min(zoom_val))
        best_ep   = zoom_epochs[best_idx]
        best_val  = zoom_val[best_idx]
        plt.axvline(best_ep, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best val (ep {best_ep}, loss={best_val:.4f})")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Training & Validation Loss (zoomed, epoch 50+)", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path_zoom, dpi=300)
        plt.show()
        print(f"Zoomed plot saved to: {save_path_zoom}")

if __name__ == "__main__":
    json_path = "/home/b10502010/work/GuitarTab/ckpt/v2_aux/training_log.json"
    plot(json_path)