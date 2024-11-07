import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torchvision.transforms import v2
from utils.dataloader import Data
from args import args
from sum_stats import mean_std, min_max
from models.Pix2Pix import Generator
from models.EncoderDecoder import StressHeightAE
from models.UNet import StressHeightUNet
from utils.helpers import load_model, save_comparison_plot, save_pressure_surface_plot

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if args.model == "pix2pix":
    dataset = Data(args.path, augment=True, scale=False, min_max=min_max, tanh=True, preload=True, img_size=(args.img_size_w, args.img_size_h))
elif args.model == "encoderdecoder" or args.model == "unet":
    dataset = Data(args.path, augment=True, scale=False, min_max=min_max, preload=True, img_size=(args.img_size_w, args.img_size_h))
else:
    print("Please parse 'encoderdecoder', 'unet' or 'pix2pix'.")
    exit()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

print("Samples in Testingset:", len(test_loader.dataset))

exp_id = np.random.randint(1, 10_000)

if args.model == "pix2pix":
    G = Generator(input_dim=2, num_filter=64, output_dim=1).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    path = f"saved_models/{exp_id}/G.tar"
    model = load_model(path, G, G_optimizer)

elif args.model == "encoderdecoder":
    encdec = StressHeightAE(z_dim=64, z_w=8, z_h=16, conv_dim=64, dropout=0.0).to(device)
    encdec_optimizer = torch.optim.Adam(encdec.parameters(), lr=5e-5)
    path = f"saved_models/{exp_id}/encoder_decoder.tar"
    model = load_model(path, encdec, encdec_optimizer)

elif args.model == "unet":
    unet = StressHeightUNet().to(device)
    unet_optimizer = torch.optim.Adam(unet.parameters(), lr=5e-5)
    path = f"saved_models/{exp_id}/u_net.tar"
    model = load_model(path, unet, unet_optimizer)

rnd_idx = np.random.randint(0, len(test_data))
print(rnd_idx)
src_data = test_data[rnd_idx]

model.eval()
with torch.no_grad():
    x_input = torch.cat((src_data[0].unsqueeze(0).unsqueeze(1).to(device), src_data[2].unsqueeze(0).unsqueeze(1).to(device)), dim=1)
    if args.model == "pix2pix":
        pred = model(x_input)
    else:
        pred, z = model(x_input)
    pred = pred.squeeze().detach().cpu().numpy()

pred = ((pred + 1) / 2) * (3 - 1) + 1
true = ((src_data[1] + 1) / 2) * (3 - 1) + 1

if args.model == "pix2pix":
    comp_path = "output/prediction_pix2pix.png"
elif args.model == "encoderdecoder":
    comp_path = "output/prediction_encoder_decoder.png"
elif args.model == "unet":
    comp_path = "output/prediction_u_net.png"
save_comparison_plot(comp_path, true, pred)

pressure_die = src_data[0]  * min_max[1]
pressure_punch = src_data[2]  * min_max[-1]
surface_die = src_data[1] * 2 + 1

ps_path = "output/pressure_surface.png"

save_pressure_surface_plot(ps_path, pressure_die, pressure_punch, surface_die)

transforms = v2.Compose([v2.GaussianNoise(mean=0.0, sigma=args.inject_noise, clip=True)])

mse = MSELoss()
mae = L1Loss()

model.eval()
mse_loss = 0.0
mae_loss = 0.0
size_loader = len(test_loader.dataset)
with torch.no_grad():
    for stress, points, punch, condition in tqdm(test_loader):
        points = points.to(device)
        x_input = torch.cat((stress.unsqueeze(1), punch.unsqueeze(1)), dim=1).to(device)
        if args.inject_noise > 0.0:
            x_input = transforms(x_input)
        if args.model == "pix2pix":
            pred_points = model(x_input).squeeze(1)
        else:
            pred_points, z = model(x_input)
        mse_loss += mse(pred_points, points).item() * stress.size(0)
        mae_loss += mae(pred_points, points).item() * stress.size(0)

    mse_loss = mse_loss / size_loader
    mae_loss = mae_loss / size_loader

    print(f"MSE: {mse_loss:.5f}")
    print(f"MAE: {mae_loss:.5f}")

if args.inject_noise > 0.0:
    ps_path = f"output/noise_{args.inject_noise}_pressure_surface.png"
    save_pressure_surface_plot(ps_path, transforms(pressure_die), transforms(pressure_punch), surface_die)
