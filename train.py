import torch
import os
import numpy as np
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.dataloader import Data
from args import args
from sum_stats import mean_std, min_max
from models.Pix2Pix import Generator, Discriminator
from models.EncoderDecoder import StressHeightAE
from models.UNet import StressHeightUNet
from models.train_test import train, test, train_pix2pix, test_pix2pix
from utils.helpers import save_best_model, save_plot

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if args.model == "pix2pix":
    dataset = Data(args.path, augment=True, scale=True, min_max=min_max, tanh=True, preload=True, img_size=(args.img_size_w, args.img_size_h))
else:
    dataset = Data(args.path, augment=True, scale=True, min_max=min_max, preload=True, img_size=(args.img_size_w, args.img_size_h))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
print("Samples in Trainingset:", len(train_loader.dataset))
print("Samples in Testingset:", len(test_loader.dataset))

exp_id = np.random.randint(1, 10_000)
print("ID: ", exp_id)

try:
    os.mkdir(f"saved_models/{exp_id}/")
except FileExistsError:
    pass

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

    train_hist, test_hist = [], []

    best_loss = np.inf

    path = f"saved_models/{exp_id}/G.tar"

    for epoch in tqdm(range(1, args.epochs + 1)):
        D_avg_loss, G_avg_loss = train_pix2pix(G, D, G_optimizer, D_optimizer, train_loader, BCE_loss, L1_loss, lambda_pixel)
        avg_test_loss = test_pix2pix(G, D, test_loader, MSE_loss)
        train_hist.append([D_avg_loss, G_avg_loss])
        test_hist.append([avg_test_loss])
        print("Epoch [%d/%d], D_loss: %.4f, G_loss: %.4f / Testdata: MSE_loss : %.4f" % ( epoch, args.epochs, D_avg_loss, G_avg_loss, avg_test_loss))
        best_loss = save_best_model(avg_test_loss, best_loss, epoch, path, G, G_optimizer)

        plt.figure(figsize=(20, 10))
        plt.plot(np.array(train_hist)[:, 1], label="Generator (Train)")
        plt.plot(np.array(train_hist)[:, 0], label="Discriminator (Train)")
        plt.plot(np.array(test_hist), label="Generator (Test)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"saved_models/{exp_id}/Pix2Pix_loss.png")

elif args.model == "encoderdecoder":
    encdec = StressHeightAE(z_dim=64, z_w=8, z_h=16, conv_dim=64, dropout=0).to(device)

    encdec_optimizer = torch.optim.Adam(encdec.parameters(), lr=5e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 4, gamma=0.5)

    loss_func = torch.nn.MSELoss()

    best_loss = np.inf

    train_hist, test_hist = [], []

    path = f"saved_models/{exp_id}/encoder_decoder.tar"

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train(encdec, encdec_optimizer, train_loader, loss_func)
        test_loss = test(encdec, test_loader, loss_func)
        train_hist.append([train_loss])
        test_hist.append([test_loss])

        print(f"Epoch: {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        best_loss = save_best_model(test_loss, best_loss, epoch, path, encdec, encdec_optimizer)

        fig_path = f"saved_models/{exp_id}/Encoder_Decoder_loss.png"
        save_plot(fig_path, train_hist, test_hist)

        # scheduler.step()

elif args.model == "unet":
    unet = StressHeightUNet().to(device)

    unet_optimizer = torch.optim.Adam(unet.parameters(), lr=5e-5)

    loss_func = torch.nn.MSELoss()

    best_loss = np.inf

    train_hist, test_hist = [], []

    path = f"saved_models/{exp_id}/u_net.tar"

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train(unet, unet_optimizer, train_loader, loss_func)
        test_loss = test(unet, test_loader, loss_func)
        train_hist.append([train_loss])
        test_hist.append([test_loss])

        print(f"Epoch: {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

        best_loss = save_best_model(test_loss, best_loss, epoch, path, unet, unet_optimizer)

        fig_path = f"saved_models/{exp_id}/U_Net_loss.png"
        save_plot(fig_path, train_hist, test_hist)
else:
    print("Please parse 'encoderdecoder', 'unet' or 'pix2pix'.")
    exit()