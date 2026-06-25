"""YOLO-v1 style detector for PASCAL VOC.

A ResNet-50 backbone (pretrained on ImageNet) feeds a small convolutional head that outputs an
S x S x (B*5 + C) tensor, following the YOLO-v1 prediction format. A pretrained backbone is used so
the detector reaches a usable mAP within a practical number of training epochs.
"""
import torch
import torch.nn as nn
import torchvision


class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20, pretrained=True):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet50(weights=weights)
        # keep everything up to (and including) layer4 -> 2048 x 14 x 14 for a 448 input
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, stride=2, padding=1), nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(0.5),                       # 14 -> 7
            nn.Conv2d(1024, B * 5 + C, 1))                            # -> S x S x (B*5+C)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)                       # (N, B*5+C, S, S)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, S, S, B*5+C)
        # All targets are in [0, 1] (class one-hots, confidence=IoU, cell-relative x/y,
        # image-relative w/h), so a sigmoid keeps the outputs in the same range. Without it the
        # raw linear outputs stay tiny and confidences never reach a usable scale.
        return torch.sigmoid(x)
