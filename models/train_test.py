import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, train_loader, loss_function):
    model.train()
    train_loss = 0.0
    for stress, points, punch, _ in train_loader:
        points = points.to(device)
        x_input = torch.cat((stress.unsqueeze(1), punch.unsqueeze(1)), dim=1).to(device)
        pred_points, z = model(x_input)
        loss = loss_function(pred_points, points)
        train_loss += loss.item() * x_input.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader.dataset)
    return train_loss

def test(model, test_loader, loss_function):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for stress, points, punch, _ in test_loader:
            points = points.to(device)
            x_input = torch.cat((stress.unsqueeze(1), punch.unsqueeze(1)), dim=1).to(device)
            pred_points, z = model(x_input)
            loss = loss_function(pred_points, points)
            test_loss += loss.item() * x_input.size(0)
        test_loss = test_loss / len(test_loader.dataset)
        return test_loss