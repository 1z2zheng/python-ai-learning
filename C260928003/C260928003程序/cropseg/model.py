from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet50


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionGate(nn.Module):
    """用解码器特征对跳跃连接进行空间注意力筛选。"""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.score(self.gate_projection(gate) + self.skip_projection(skip))
        return skip * attention


class AttentionUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            inter_channels=max(out_channels // 2, 1),
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attended_skip = self.attention(x, skip)
        return self.conv(torch.cat([attended_skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class MultiScaleContext(nn.Module):
    """轻量多尺度上下文模块。

    并行的 1×1 与不同膨胀率 3×3 分支同时保留小麦穗局部纹理和
    被遮挡时所需的较大范围语义线索。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = max(channels // 4, 1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
                *(
                    nn.Sequential(
                        nn.Conv2d(
                            channels, branch_channels, 3, padding=rate,
                            dilation=rate, bias=False,
                        ),
                        nn.BatchNorm2d(branch_channels),
                        nn.ReLU(inplace=True),
                    )
                    for rate in (2, 4, 6)
                ),
            ]
        )
        merged_channels = branch_channels * len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(merged_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class MultiScaleUNet(nn.Module):
    """U-Net with dilated context and full-resolution shallow-feature refinement."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.context = MultiScaleContext(c * 16)
        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)
        self.refine = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.context(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(self.refine(torch.cat([x, x1], dim=1)))


class AttentionUNet(nn.Module):
    """U-Net with additive attention gates on all four skip connections."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = AttentionUp(c * 16, c * 8, c * 8)
        self.up2 = AttentionUp(c * 8, c * 4, c * 4)
        self.up3 = AttentionUp(c * 4, c * 2, c * 2)
        self.up4 = AttentionUp(c * 2, c, c)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = self.project(self.pool(x))
        return nn.functional.interpolate(
            pooled, size=size, mode="bilinear", align_corners=False
        )


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: tuple[int, int, int] = (6, 12, 18),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                *(ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates),
                ASPPPooling(in_channels, out_channels),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(self.branches) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with a ResNet50 encoder and output stride 16."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        del base_channels  # Kept in the shared constructor signature for config compatibility.
        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = ASPP(2048, 256, (6, 12, 18))
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        low_level = self.low_level_projection(low_level)
        x = nn.functional.interpolate(
            x, size=low_level.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.decoder(torch.cat([x, low_level], dim=1))
        return nn.functional.interpolate(
            x, size=input_size, mode="bilinear", align_corners=False
        )


def normalize_model_name(name: str | None) -> str:
    normalized = (name or "unet").strip().lower().replace("-", "_")
    aliases = {
        "u_net": "unet",
        "attentionunet": "attention_unet",
        "attention_u_net": "attention_unet",
        "attunet": "attention_unet",
        "deeplabv3+": "deeplabv3plus_resnet50",
        "deeplabv3plus": "deeplabv3plus_resnet50",
        "deeplab_v3_plus": "deeplabv3plus_resnet50",
        "ms_unet": "multiscale_unet",
        "multi_scale_unet": "multiscale_unet",
    }
    return aliases.get(normalized, normalized)


def build_model(
    name: str | None,
    in_channels: int,
    num_classes: int,
    base_channels: int = 32,
) -> nn.Module:
    model_name = normalize_model_name(name)
    model_classes = {
        "unet": UNet,
        "attention_unet": AttentionUNet,
        "deeplabv3plus_resnet50": DeepLabV3Plus,
        "multiscale_unet": MultiScaleUNet,
    }
    if model_name not in model_classes:
        supported = ", ".join(sorted(model_classes))
        raise ValueError(f"不支持的模型 {name!r}，可选：{supported}")
    return model_classes[model_name](in_channels, num_classes, base_channels)

