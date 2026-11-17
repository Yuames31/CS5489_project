"""
Segmentation architectures.

``ResNetUNet`` -- the model used for the final submission:

    ResNet-34/50 (ImageNet) encoder
      -> ASPP head on the 1/32 feature
      -> 4 decoder stages with skip connections and SCSE attention
      -> full-resolution refinement against a light detail branch of the input
      -> main logits at 1/1 + deep-supervision logits at 1/2 and 1/4

Why this and not plain U-Net:
  * the edited regions are *objects*, so an ImageNet-pretrained encoder gives a
    much better prior than random init (5000 images is small);
  * AIGC masks follow object boundaries closely, so the 1/2 skip and the
    full-resolution detail branch matter -- without them the last bilinear
    upsample blurs the boundary and costs several IoU points;
  * SCSE re-weights channels/spatial positions in the decoder, which suppresses
    the large photographic background;
  * deep supervision stabilises training of the deep decoder.

``--hf`` additionally feeds a high-pass residual (x - mean3x3(x)) into the stem,
which makes the low-level "noise fingerprint" of the generator explicitly
available.  This is a cheap, well-known forgery-detection trick and is provided
as an ablation switch (off by default).

``DeepLabV3Plus`` wraps the torchvision implementation so it can be used as an
ablation baseline with exactly the same training loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet as tv_resnet

__all__ = ["build_model", "ResNetUNet", "DeepLabV3Plus", "MODEL_OUTPUTS"]

MODEL_OUTPUTS = ("main", "aux_half", "aux_quarter")


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, d=1, groups=1):
        super().__init__()
        p = d * (k - 1) // 2 if p is None else p
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, dilation=d, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SCSE(nn.Module):
    """Concurrent spatial and channel squeeze-and-excitation."""

    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(ch // r, 4)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1), nn.Sigmoid())
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class ASPP(nn.Module):
    def __init__(self, cin, cout=256, dilations=(1, 2, 4)):
        super().__init__()
        branches = [ConvBNAct(cin, cout, 1)]
        for d in dilations:
            branches.append(ConvBNAct(cin, cout, 3, p=d, d=d))
        branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # no BatchNorm here: with batch_size=1 this is a single value per
            # channel and BN would raise
            nn.Conv2d(cin, cout, 1, bias=True), nn.ReLU(inplace=True)))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            ConvBNAct(cout * len(branches), cout, 1), nn.Dropout2d(0.1))

    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) if i < len(self.branches) - 1 else
                 F.interpolate(b(x), size=size, mode="bilinear", align_corners=False)
                 for i, b in enumerate(self.branches)]
        return self.project(torch.cat(feats, 1))


class DecoderBlock(nn.Module):
    """Upsample + concat skip + conv, with SCSE attention."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(cin + cskip, cout, 3),
            ConvBNAct(cout, cout, 3),
            SCSE(cout),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], 1))


# --------------------------------------------------------------------------- #
# ResNet U-Net
# --------------------------------------------------------------------------- #
_RESNET_TABLE = {
    "resnet18": (tv_resnet.resnet18, 64, [64, 128, 256, 512], "ResNet18_Weights"),
    "resnet34": (tv_resnet.resnet34, 64, [64, 128, 256, 512], "ResNet34_Weights"),
    "resnet50": (tv_resnet.resnet50, 64, [256, 512, 1024, 2048], "ResNet50_Weights"),
}


def _load_backbone(arch, pretrained):
    ctor, _, _, wname = _RESNET_TABLE[arch]
    if pretrained:
        try:
            weights = getattr(torchvision.models, wname).IMAGENET1K_V1
            net = ctor(weights=weights)
            print(f"[model] loaded ImageNet weights for {arch}")
            return net
        except Exception as exc:                      # offline / no cache
            print(f"[model] WARNING: no pretrained weights ({exc}); random init")
    return ctor(weights=None)


class ResNetUNet(nn.Module):
    def __init__(self, arch="resnet34", pretrained=True, hf=False,
                 base=32, deep_sup=True):
        super().__init__()
        assert arch in _RESNET_TABLE, f"unknown encoder {arch}"
        self.arch = arch
        self.deep_sup = deep_sup
        self.hf = hf

        bb = _load_backbone(arch, pretrained)
        ch = _RESNET_TABLE[arch][2]                    # [c1, c2, c3, c4] channels

        self.stem_conv = bb.conv1
        self.stem_bn = bb.bn1
        self.stem_act = bb.relu
        self.maxpool = bb.maxpool
        self.layer1, self.layer2 = bb.layer1, bb.layer2
        self.layer3, self.layer4 = bb.layer3, bb.layer4

        if hf:                                          # 3 RGB + 3 high-pass
            old = self.stem_conv
            new = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride,
                            old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                new.weight[:, 3:] = 0.0
                if old.bias is not None:
                    new.bias[:] = old.bias
            self.stem_conv = new

        c4, c3, c2, c1 = ch[3], ch[2], ch[1], ch[0]
        self.aspp = ASPP(c4, cout=max(base * 8, 256))
        self.dec4 = DecoderBlock(max(base * 8, 256), c3, base * 8)
        self.dec3 = DecoderBlock(base * 8, c2, base * 4)
        self.dec2 = DecoderBlock(base * 4, c1, base * 2)
        self.dec1 = DecoderBlock(base * 2, c1, base)

        self.detail = ConvBNAct(3, base // 2, 3)        # full-resolution branch
        self.final = nn.Sequential(
            ConvBNAct(base + base // 2, base, 3), ConvBNAct(base, base, 3))
        self.head = nn.Conv2d(base, 1, 1)
        self.aux_half = nn.Conv2d(base, 1, 1)
        self.aux_quarter = nn.Conv2d(base * 2, 1, 1)

    def forward(self, x):
        if self.hf:
            hf = x - F.avg_pool2d(x, 3, 1, 1)
            x_in = torch.cat([x, hf], 1)
            detail_src = x
        else:
            x_in = x
            detail_src = x

        e1 = self.stem_act(self.stem_bn(self.stem_conv(x_in)))     # 1/2
        e2 = self.layer1(self.maxpool(e1))                          # 1/4
        e3 = self.layer2(e2)                                        # 1/8
        e4 = self.layer3(e3)                                        # 1/16
        e5 = self.layer4(e4)                                        # 1/32

        d4 = self.dec4(self.aspp(e5), e4)                           # 1/16
        d3 = self.dec3(d4, e3)                                      # 1/8
        d2 = self.dec2(d3, e2)                                      # 1/4
        s1 = self.dec1(d2, e1)                                      # 1/2

        d0 = F.interpolate(s1, scale_factor=2, mode="bilinear", align_corners=False)
        d0 = self.final(torch.cat([d0, self.detail(detail_src)], 1))  # 1/1
        main = self.head(d0)

        if not self.deep_sup:
            return [main]
        return [main, self.aux_half(s1), self.aux_quarter(d2)]


# --------------------------------------------------------------------------- #
# torchvision DeepLabV3+ (ablation baseline)
# --------------------------------------------------------------------------- #
class DeepLabV3Plus(nn.Module):
    def __init__(self, pretrained=True, **_):
        super().__init__()
        try:
            wb = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=None, weights_backbone=wb, num_classes=1,
                aux_loss=True)
            print("[model] built torchvision DeepLabV3-ResNet50")
        except Exception as exc:
            print(f"[model] WARNING: DeepLabV3 fallback ({exc})")
            net = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=None, weights_backbone=None, num_classes=1, aux_loss=True)
        self.net = net
        self.deep_sup = True

    def forward(self, x):
        out = self.net(x)
        main = out["out"]
        aux = out.get("aux")
        if aux is None:
            return [main]
        return [main, aux]


def build_model(arch="resnet34", pretrained=True, hf=False, base=32, deep_sup=True):
    if arch.startswith("deeplab"):
        return DeepLabV3Plus(pretrained=pretrained)
    return ResNetUNet(arch=arch, pretrained=pretrained, hf=hf,
                      base=base, deep_sup=deep_sup)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
