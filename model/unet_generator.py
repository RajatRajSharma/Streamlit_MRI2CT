# model/unet_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.enc1 = self.encoder_block(input_channels, 64)  # 256x256 -> 128x128
        self.enc2 = self.encoder_block(64, 128)  # 128x128 -> 64x64
        self.enc3 = self.encoder_block(128, 256)  # 64x64 -> 32x32
        self.enc4 = self.encoder_block(256, 512)  # 32x32 -> 16x16

        # Bottleneck
        self.bottleneck = self.encoder_block(512, 1024)  # 16x16 -> 8x8

        # Decoder
        self.dec4 = self.decoder_block(1024, 512)  # 8x8 -> 16x16
        self.dec3 = self.decoder_block(1024, 256)  # 16x16 -> 32x32
        self.dec2 = self.decoder_block(512, 128)  # 32x32 -> 64x64
        self.dec1 = self.decoder_block(256, 64)  # 64x64 -> 128x128

        # Final layer to restore the original spatial size
        self.final_upsample = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256
        
        # Activation function
        self.tanh = nn.Tanh()

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.14, inplace=True),
            nn.Dropout(p=0.1)
        )
    
    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # 256x256 -> 128x128
        enc2 = self.enc2(enc1)  # 128x128 -> 64x64
        enc3 = self.enc3(enc2)  # 64x64 -> 32x32
        enc4 = self.enc4(enc3)  # 32x32 -> 16x16

        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # 16x16 -> 8x8

        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        
        # Final upsampling and activation
        out = self.final_upsample(dec1)
        return self.tanh(out).view(-1, 1, 256, 256)