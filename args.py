import argparse

parser = argparse.ArgumentParser(description="Pressure Distribution")
parser.add_argument("--path", default="data/Geometrien", help="Path of dataset.", type=str)
parser.add_argument("--epochs", default=1000, help="Number of epochs.", type=int)
parser.add_argument("--batch_size", default=16, help="Number of batch size.", type=int)
parser.add_argument("--num_workers", default=0, help="Number of workers.", type=int)
parser.add_argument("--seed", default=42, help="Seed for reproducibility.", type=int)
parser.add_argument("--img_size_w", default=256, help="Width of height map.", type=int)
parser.add_argument("--img_size_h", default=128, help="Height of height map.", type=int)
parser.add_argument("--model", default="encoderdecoder", help="Parse 'encoderdecoder', 'unet' or 'pix2pix'.", type=str)
parser.add_argument("--inject_noise", default=0.0, help="Level of injected noise (0.01 for 1% noise).", type=float)
args = parser.parse_args()