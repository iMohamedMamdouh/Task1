import torch
from torch import nn
from torchvision.models import resnet18

# download.pytorch.org is unreachable from this environment, the legacy S3
# bucket serves the same torchvision ImageNet checkpoint.
IMAGENET_URL = "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth"


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNetResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        encoder = resnet18()
        if pretrained:
            state = torch.hub.load_state_dict_from_url(IMAGENET_URL, progress=False)
            encoder.load_state_dict(state)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.up4 = conv_block(512 + 256, 256)
        self.up3 = conv_block(256 + 128, 128)
        self.up2 = conv_block(128 + 64, 64)
        self.up1 = conv_block(64 + 64, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        d4 = self.up4(torch.cat([upsample_to(s4, s3), s3], 1))
        d3 = self.up3(torch.cat([upsample_to(d4, s2), s2], 1))
        d2 = self.up2(torch.cat([upsample_to(d3, s1), s1], 1))
        d1 = self.up1(torch.cat([upsample_to(d2, s0), s0], 1))
        return upsample_to(self.head(d1), x)


def upsample_to(x, reference):
    return nn.functional.interpolate(
        x, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )
