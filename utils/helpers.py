import torch
import matplotlib.pyplot as plt
import numpy as np

def save_best_model(avg_test_loss, best_loss, epoch, path, model, optimizer):
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        print(20 * "#")
        print("Best model at epoch: %i, with loss: %.6f" % (epoch, best_loss))
        print(20 * "#")
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    }, path)
    return best_loss

def load_model(path, model, optimizer, eval=True):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("loaded at epoch: %i, with loss: %.6f" % (epoch, loss))

    if eval:
        model.eval()
        return model
    else:
        return model, optimizer

def save_plot(path, train_history, test_history):
    plt.figure(figsize=(20, 10))
    plt.plot(np.array(train_history), label="Train")
    plt.plot(np.array(test_history), label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()

def save_pressure_surface_plot(path, pressure_die, pressure_punch, surface_die):
    vmin = min(pressure_die.min(), pressure_punch.min())
    vmax = max(pressure_die.max(), pressure_punch.max())

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(pressure_die, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Pressure (MPa)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Pressure Distribution of Matrix")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(surface_die, cmap="jet")
    plt.colorbar(label="z (mm)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Surface of Matrix")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pressure_punch, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Pressure (MPa)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Pressure Distribution of Punch")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_comparison_plot(path, true, pred):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(true, cmap="jet")
    plt.colorbar(label="z (mm)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Target: Surface of Matrix")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="jet")
    plt.colorbar(label="z (mm)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Prediction: Surface of Matrix")
    plt.axis("off")

    diff = np.abs(pred - np.array(true))
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="jet")
    plt.colorbar(label="z (mm)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()