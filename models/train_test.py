import torch
from args import args

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

def train_pix2pix(G, D, G_optimizer, D_optimizer, train_loader, BCE_loss, L1_loss, lambda_pixel):
    train_loader_size = (len(train_loader.dataset) - len(train_loader.dataset) % args.batch_size)

    D_train_loss = 0.0
    G_train_loss = 0.0

    G.train()
    D.train()
    for stress, points, punch, _ in train_loader:
            x_input = torch.cat((stress.unsqueeze(1), punch.unsqueeze(1)), dim=1).to(device)
            y_target = points.unsqueeze(1).to(device)
            # Train discriminator with real data
            D_real_decision = D(x_input, y_target).squeeze()
            real_ = torch.ones(D_real_decision.size()).to(device)
            D_real_loss = BCE_loss(D_real_decision, real_)
            # Train discriminator with fake data
            gen_image = G(x_input)
            D_fake_decision = D(x_input, gen_image).squeeze()
            fake_ = torch.zeros(D_fake_decision.size()).to(device)
            D_fake_loss = BCE_loss(D_fake_decision, fake_)
            # Combined D loss
            D_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss += D_loss.item() * x_input.size(0)
            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            # Train generator
            gen_image = G(x_input)
            D_fake_decision = D(x_input, gen_image).squeeze()
            G_fake_loss = BCE_loss(D_fake_decision, real_)
            # L1 loss
            l1_loss = lambda_pixel * L1_loss(gen_image, y_target)
            # Combined G loss
            G_loss = G_fake_loss + l1_loss
            G_train_loss += G_loss.item() * x_input.size(0)
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

    D_avg_loss = D_train_loss / train_loader_size
    G_avg_loss = G_train_loss / train_loader_size
    return D_avg_loss, G_avg_loss

def test_pix2pix(G, D, test_loader, MSE_loss):
    test_loss = 0.0

    G.eval()
    D.eval()
    with torch.no_grad():
        for stress, points, punch, _ in test_loader:
            x_input = torch.cat((stress.unsqueeze(1), punch.unsqueeze(1)), dim=1).to(device)
            y_target = points.unsqueeze(1).to(device)
            test_image = G(x_input)
            mse = MSE_loss(y_target, test_image)
            test_loss += mse.item() * x_input.size(0)

        avg_test_loss = test_loss / len(test_loader.dataset)
        return avg_test_loss
