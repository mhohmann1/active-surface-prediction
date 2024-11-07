import torch

def save_best_model(avg_test_loss, best_loss, epoch, path, model, optimizer):
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss

        print(20 * "#")
        print("Best models at epoch: %i, with loss: %.4f" % (epoch, best_loss))
        print(20 * "#")

        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss, }, path)

        return best_loss

def load_model(path, model, optimizer, eval=True):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("loaded at epoch: ", epoch, " and mse loss: ", loss)

    if eval:
        model.eval()
        return model
    else:
        return model, optimizer