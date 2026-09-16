#unet_seg_model.py
#紧凑版 U-Net，专门给 32x40 的粘连数字图用（两次下采样：32x40 -> 16x20 -> 8x10）
#输出3类逐像素 logits: 0=背景, 1=左字符, 2=右字符
#和 cnn4032noise1dig3.py 里 CNN 的下采样倍数(4x)刻意保持一致，方便理解/调参

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """输入 (B,1,32,40) -> 输出 (B,num_classes,32,40) 逐像素 logits"""

    def __init__(self, in_ch=1, num_classes=3, base=16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)               # 32x40
        self.enc2 = ConvBlock(base, base * 2)             # 16x20
        self.bottleneck = ConvBlock(base * 2, base * 4)   # 8x10
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
