import argparse

parser = argparse.ArgumentParser(description="Pressure Distribution")
parser.add_argument("--path", default="data/Geometrien", help="Path of dataset.", type=str)
parser.add_argument("--epochs", default=100, help="Number of epochs.", type=int)
parser.add_argument("--batch_size", default=32, help="Number of batch size.", type=int)
parser.add_argument("--workers", default=4, help="Number of workers.", type=int)
parser.add_argument("--learning_rate", default=1e-3, help="Size of learning rate.", type=int)
parser.add_argument("--seed", default=42, help="Seed for reproducibility.", type=int)
parser.add_argument("--img_size_w", default=256, help="Width of height map.", type=int)
parser.add_argument("--img_size_h", default=128, help="Height of height map.", type=int)
args = parser.parse_args()