import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, in_features=64, depth=4):
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
        for features in reversed(range(depth - 1)):
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