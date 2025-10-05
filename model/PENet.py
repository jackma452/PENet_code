import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .Encoder import All2Cross
from .Decoder import ConvUpsample, SegmentationHead
import configs.configs as configs

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


class PENet(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config = config, img_size= img_size, in_chans=in_chans)

        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128,128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        self.cga = CGAFusion(dim=96)
        self.cga_0 = CGAFusion(dim=384)

    def forward(self, x):
        xs, res = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        embeddings_1 = [x[:, 1:] for x in res]
        reshaped_embed = []
        reshaped_embed_1 = []

        for i, embed_1 in enumerate(embeddings_1):

            embed_1 = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed_1)
            reshaped_embed_1.append(embed_1)

        for i, embed in enumerate(embeddings):

            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
            embed = self.cga(embed, reshaped_embed_1[0]) if i == 0 else self.cga_0(embed, reshaped_embed_1[1])
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)

            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        out = self.segmentation_head(C)

        return out

