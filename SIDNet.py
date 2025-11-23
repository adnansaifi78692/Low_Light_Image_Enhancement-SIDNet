import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------
# Residual Block for SIDNet
# ---------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)

# ---------------------------------------
# SIDNet Architecture
# ---------------------------------------
class SIDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False)
        self.rb1 = ResidualBlock(64)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.rb2 = ResidualBlock(128)
        self.rb_mid = ResidualBlock(128)
        self.rb_mid2 = ResidualBlock(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.rb3 = ResidualBlock(64)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.rb4 = ResidualBlock(32)
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = F.relu(self.rb1(self.down1(x0)))
        x2 = F.relu(self.rb2(self.down2(x1)))
        xm = self.rb_mid(self.rb_mid2(x2))
        y1 = F.relu(self.rb3(self.up1(xm))) + x1
        y2 = F.relu(self.rb4(self.up2(y1))) + x0
        return torch.sigmoid(self.conv_out(y2))
