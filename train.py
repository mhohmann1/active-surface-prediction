import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from utils.dataloader import Data
from args import args
from sum_stats import mean_std, min_max
from models.Pix2Pix import Generator, Discriminator
from models.EncoderDecoder import StressHeightAE
from models.UNet import StressHeightUNet

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if args.model == "pix2pix":
    dataset = Data(args.path, augment=True, scale=False, min_max=min_max, tanh=True, preload=True, img_size=(args.img_size_w, args.img_size_h))
else:
    dataset = Data(args.path, augment=True, scale=False, min_max=min_max, preload=True, img_size=(args.img_size_w, args.img_size_h))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
print("Samples in Trainingset:", len(train_loader.dataset))
print("Samples in Testingset:", len(test_loader.dataset))

exp_id = np.random.randint(1, 10_000)
print("ID: ", exp_id)

if args.model == "pix2pix":
    G = Generator(input_dim=2, num_filter=64, output_dim=1).to(device)
    D = Discriminator(input_dim=3, num_filter=64, output_dim=1).to(device)

    BCE_loss = torch.nn.BCELoss()
    L1_loss = torch.nn.L1Loss()
    MSE_loss = torch.nn.MSELoss()

    lambda_pixel = 100

    lr = 2e-4
    betas = (0.5, 0.999)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    train_loader_size = (len(train_loader.dataset) - len(train_loader.dataset) % args.batch_size)

    train_hist, test_hist = [], []

    best_loss = np.inf

    EPOCHS = 1_000

    for epoch in tqdm(range(1, EPOCHS + 1)):

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

            D_train_loss += D_loss.item() * points.size(0)

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

            G_train_loss += G_loss.item() * points.size(0)

            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

        D_avg_loss = D_train_loss / train_loader_size
        G_avg_loss = G_train_loss / train_loader_size
        train_hist.append([D_avg_loss, G_avg_loss])
        # print('Epoch [%d/%d], D_loss: %.4f, G_loss: %.4f' % (epoch, EPOCHS, D_avg_loss, G_avg_loss))

        test_loss = 0.0

        G.eval()
        D.eval()
        with torch.no_grad():
            for stress, points, punch, _ in train_loader:
                x_input = torch.cat((stress.unsqueeze(1), punch.unsqueeze(1)), dim=1).to(device)
                y_target = points.unsqueeze(1).to(device)
                test_image = G(x_input)
                mse = MSE_loss(y_target, test_image)
                test_loss += mse.item() * points.size(0)

            avg_test_loss = test_loss / len(test_loader.dataset)
            test_hist.append([avg_test_loss])

            print('Epoch [%d/%d], D_loss: %.4f, G_loss: %.4f / Testdata: MSE_loss : %.4f' % (
            epoch, EPOCHS, D_avg_loss, G_avg_loss, avg_test_loss))

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss

                print(20 * "#")
                print("Best models at epoch: %i, with loss: %.4f" % (epoch, best_loss))
                print(20 * "#")

                PATH = f"G_ID_{exp_id}.tar"

                torch.save({"epoch": epoch,
                            "model_state_dict": G.state_dict(),
                            "optimizer_state_dict": G_optimizer.state_dict(),
                            "loss": best_loss, }, PATH)

elif args.parse == "encoderdecoder":
    ae = StressHeightAE(z_dim=64, z_w=8, z_h=16, conv_dim=64, dropout=0).to(device)
    # ae = StressHeightUNet().to(device)

    EPOCH = 2_000

    optimizer = torch.optim.Adam(ae.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

    loss_fun = torch.nn.MSELoss()


    best_loss = np.inf

    train_hist, test_hist = [], []

    for epoch in tqdm(range(1, EPOCH + 1)):
        ae.train()
        train_loss = 0.0
        for stress, points, punch, condition in train_loader:
            stress = stress.unsqueeze(1).to(device)
            points = points.to(device)
            punch = punch.unsqueeze(1).to(device)
            x_input = torch.cat((stress, punch), dim=1)
            pred_points, z = ae(x_input)
            loss = loss_fun(pred_points, points)
            train_loss += loss.item() * stress.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_hist.append([train_loss])

        ae.eval()
        test_loss = 0.0
        with torch.no_grad():
            for stress, points, punch, condition in test_loader:
                stress = stress.unsqueeze(1).to(device)
                points = points.to(device)
                punch = punch.unsqueeze(1).to(device)
                x_input = torch.cat((stress, punch), dim=1)
                pred_points, z = ae(x_input)
                loss = loss_fun(pred_points, points)
                test_loss += loss.item() * stress.size(0)
            test_loss = test_loss / len(test_loader.dataset)
            test_hist.append([test_loss])
            print(f"Epoch: {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")


            if test_loss < best_loss:
                best_loss = test_loss

                print(20 * "#")
                print("Best models at epoch: %i, with loss: %.4f" % (epoch, best_loss))
                print(20 * "#")

                PATH = f"model_ID_{exp_id}.tar"
                LOSS = train_hist[-1:][0]

                torch.save({"epoch": epoch,
                            "model_state_dict": ae.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": LOSS, }, PATH)

        scheduler.step()

elif args.model == "unet":
    pass
else:
    print("Please parse 'encoderdecoder', 'unet' or 'pix2pix'.")
    exit()