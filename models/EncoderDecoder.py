import torch.nn as nn

class ImgEncoder(nn.Module):
    def __init__(self, z_dim, z_w, z_h, conv_dim, dropout):
        super(ImgEncoder, self).__init__()

        self.z_w = z_w
        self.z_h = z_h
        self.conv_dim = conv_dim

        self.enconv = nn.Sequential(
            nn.Conv2d(2, self.conv_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.conv_dim),
            nn.Dropout2d(dropout),
            nn.ReLU(),

            nn.Conv2d(self.conv_dim, self.conv_dim*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.conv_dim*2),
            nn.ReLU(),

            nn.Conv2d(self.conv_dim*2, self.conv_dim*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.conv_dim*4),
            nn.ReLU(),

            nn.Conv2d(self.conv_dim*4, self.conv_dim*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.conv_dim*8),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.conv_dim*8 * self.z_w * self.z_h, z_dim),
        )

    def forward(self, x):
        x = self.enconv(x)
        x = x.view(-1, self.conv_dim*8 * self.z_w * self.z_h)
        x = self.fc(x)
        return x


class HeightMapDecoder(nn.Module):
    def __init__(self, z_dim, z_w, z_h, conv_dim):
        super(HeightMapDecoder, self).__init__()

        self.z_w = z_w
        self.z_h = z_h

        self.conv_dim = conv_dim

        self.fc = nn.Sequential(
            nn.Linear(z_dim, self.conv_dim*8 * self.z_w * self.z_h),
            nn.BatchNorm1d(self.conv_dim*8 * self.z_w * self.z_h),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.conv_dim*8, self.conv_dim*4, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(self.conv_dim*4),
            nn.ReLU(),

            nn.ConvTranspose2d(self.conv_dim*4, self.conv_dim*2, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(self.conv_dim*2),
            nn.ReLU(),

            nn.ConvTranspose2d(self.conv_dim*2, self.conv_dim, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(self.conv_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(self.conv_dim, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, self.conv_dim*8, self.z_w, self.z_h)
        x_hat = self.deconv(z)
        return x_hat.squeeze(1)

class StressHeightAE(nn.Module):
    def __init__(self, z_dim, z_w, z_h, conv_dim, dropout):
        super(StressHeightAE, self).__init__()
        self.encoder = ImgEncoder(z_dim, z_w, z_h, conv_dim, dropout)
        self.decoder = HeightMapDecoder(z_dim, z_w, z_h, conv_dim)

    def forward(self, img):
        z = self.encoder(img)
        x_hat = self.decoder(z)
        return x_hat, z