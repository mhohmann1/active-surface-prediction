import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from utils.dataloader import Data
from args import args

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Data(args.path, augment=False, scale=False, preload=True, img_size=(args.img_size_w, args.img_size_h))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=False)

total_sum_n = 0
total_sum_squares_n = 0
total_count_n = 0
min_val_n = float("inf")
max_val_n = float("-inf")

total_sum_m = 0
total_sum_squares_m = 0
total_count_m = 0
min_val_m = float("inf")
max_val_m = float("-inf")

total_sum_l = 0
total_sum_squares_l = 0
total_count_l = 0
min_val_l = float("inf")
max_val_l = float("-inf")

for n, m, l, _ in train_loader:
    total_sum_n += n.sum().item()
    total_sum_squares_n += (n ** 2).sum().item()
    total_count_n += n.numel()
    min_val_n = min(min_val_n, n.min().item())
    max_val_n = max(max_val_n, n.max().item())

    total_sum_m += m.sum().item()
    total_sum_squares_m += (m ** 2).sum().item()
    total_count_m += m.numel()
    min_val_m = min(min_val_m, m.min().item())
    max_val_m = max(max_val_m, m.max().item())

    total_sum_l += l.sum().item()
    total_sum_squares_l += (l ** 2).sum().item()
    total_count_l += l.numel()
    min_val_l = min(min_val_l, l.min().item())
    max_val_l = max(max_val_l, l.max().item())

mean_n = total_sum_n / total_count_n
std_n = (total_sum_squares_n / total_count_n - mean_n ** 2) ** 0.5

mean_m = total_sum_m / total_count_m
std_m = (total_sum_squares_m / total_count_m - mean_m ** 2) ** 0.5

mean_l = total_sum_l / total_count_l
std_l = (total_sum_squares_l / total_count_l - mean_l ** 2) ** 0.5

mean_std_n = (mean_n, std_n)
mean_std_m = (mean_m, std_m)
mean_std_l = (mean_l, std_l)

min_max_n = (min_val_n, max_val_n)
min_max_m = (min_val_m, max_val_m)
min_max_l = (min_val_l, max_val_l)

print(f"mean_std = {mean_std_n + mean_std_m + mean_std_l}")
print(f"min_max = {min_max_n + min_max_m + min_max_l}")