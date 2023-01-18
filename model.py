import math

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from transformers import SwinModel

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, in_features=32, depth=3):
        super(Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # downstream
        for features in range(depth):
            feature = in_features * 2 ** features
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature
            
        # upstream
        for features in reversed(range(depth)):
            feature = in_features * 2 ** features
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature*2, feature))

        # bottleneck
        self.bottleneck = ConvBlock(in_channels, in_channels*2)

        self.output = nn.Conv2d(in_features, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]

            if x.shape != skip.shape:
                x = F.resize(x, size=skip.shape[2:])

            concat = torch.concat((skip, x), dim=1)
            x = self.ups[idx+1](concat)

        return self.output(x)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, in_features=32, kernel_size=2, stride=2, depth=3):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for d in range(depth):
            features = in_features * 2 ** d
            self.encoder.append(nn.Conv2d(in_channels, features, kernel_size, padding="same"))
            self.encoder.append(nn.ReLU())
            #if d != depth - 1:
                #self.encoder.append(nn.MaxPool2d(kernel_size, stride))
            self.encoder.append(nn.MaxPool2d(kernel_size, stride))
            in_channels = features

        self.bottleneck = ConvBlock(in_channels, in_channels*2)

        for d in reversed(range(depth)):
            features = in_features * 2 ** d
            self.decoder.append(nn.ConvTranspose2d(features*2, features, kernel_size, stride))
            self.decoder.append(nn.ReLU())

        self.decoder.append(nn.ConvTranspose2d(features, out_channels, 1))
        self.decoder.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        x = self.bottleneck(x)

        for layer in self.decoder:
            x = layer(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class CustomSwinSegModel(nn.Module):
    """Creates a custom model."""

    def __init__(self, checkpoint, upscale_factor=32):
        super(CustomSwinSegModel, self).__init__()

        swin = SwinModel.from_pretrained(checkpoint)
        swin.pooler = Identity()

        embed_dim = swin.config.hidden_size
        num_layers = swin.config.num_hidden_layers
        num_features = int(embed_dim * 2 ** (num_layers - 2))

        self.model = swin

        self.decoder = nn.Sequential(nn.Conv2d(embed_dim, num_features, kernel_size=(1, 1), stride=(1, 1)),
                                     nn.PixelShuffle(upscale_factor=upscale_factor))

    def forward(self, x):
        x = self.model(x).last_hidden_state

        x = x.transpose(1, 2)
        batch_size, num_channels, sequence_length = x.shape
        height = width = math.floor(sequence_length ** 0.5)
        x = x.reshape(batch_size, num_channels, height, width)

        x = self.decoder(x)

        return x


def build_model(model_type):
    """Build a tiny Swin-Transformer.
    Args:
        model_name (str): The name of the model to build.
    Returns:
        model (nn.Module): The model.
    """
    cp = "microsoft/swin-tiny-patch4-window7-224"
    if model_type == "denoising":
        model = CustomSwinSegModel(checkpoint=cp)
    else:
        raise ValueError("Invalid model type")

    return model


if __name__ == "__main__":
    model = Unet(in_channels=3, out_channels=3, depth=3)
    x = torch.randn((1, 3, 512, 512))
    print(model(x).shape)

    p = 0
    for param in model.parameters():
        p += param.numel()
    print(f"Number of parameters: {p}")