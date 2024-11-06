import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(encoder_block, self).__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(decoder_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)

        # diffY = skip.size()[2] - x.size()[2]
        # diffX = skip.size()[3] - x.size()[3]
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        # diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x

class UNetEncoder(nn.Module):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        self.enc1 = encoder_block(2, 16)
        self.enc2 = encoder_block(16, 32)
        # self.enc3 = encoder_block(512, 1024)
        # self.enc4 = encoder_block(64, 128)
        self.z = conv_block(32, 64)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        # s3, p3 = self.enc3(p2)
        # s4, p4 = self.enc4(p3)
        z = self.z(p2)
        return s1, s2, z

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        self.dec1 = decoder_block(64, 32)
        self.dec2 = decoder_block(32, 16)
        # self.dec3 = decoder_block(64, 32)
        # self.dec4 = decoder_block(32, 16)
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=1, padding=0)

    def forward(self, s1, s2, z):
        d1 = self.dec1(z, s2)
        d2 = self.dec2(d1, s1)
        # d3 = self.dec3(d2, s2)
        # d4 = self.dec4(d3, s1)
        x_hat = self.dec3(d2)
        return x_hat

class StressHeightUNet(nn.Module):
    def __init__(self):
        super(StressHeightUNet, self).__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()

    def forward(self, img):
        s1, s2, z = self.encoder(img)
        x_hat = self.decoder(s1, s2, z)
        return x_hat.squeeze(1), z

