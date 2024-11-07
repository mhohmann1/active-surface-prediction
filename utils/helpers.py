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
