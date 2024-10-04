import torch
import torch.nn as nn
from typing import List, Tuple
import json
import os


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, shortcut: bool = True
    ) -> None:
        super(Bottleneck, self).__init__()
        self.cv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 1) -> None:
        super(C2f, self).__init__()
        hidden_channels = int(out_channels / 2)
        self.cv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0
        )
        self.cv2 = Conv(
            hidden_channels * (1 + n), out_channels, kernel_size=1, stride=1, padding=0
        )
        self.m = nn.ModuleList(
            Bottleneck(hidden_channels, hidden_channels) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.m[0](self.cv1(x)))
        return self.cv2(torch.cat([x] + y, 1))


class SPPF(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> None:
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0
        )
        self.cv2 = Conv(
            hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Detect(nn.Module):
    def __init__(
        self, nc: int = 80, anchors: Tuple[int, ...] = (), ch: Tuple[int, ...] = ()
    ) -> None:
        super(Detect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.stride = None
        self.anchors = nn.Parameter(torch.tensor(anchors).float().view(-1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * len(anchors), 1) for x in ch)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        z = []
        for i in range(len(x)):
            z.append(self.m[i](x[i]))
        return z


class YOLOv8(nn.Module):
    def __init__(self, num_classes: int = 80) -> None:
        super(YOLOv8, self).__init__()

        self.backbone = nn.Sequential(
            Conv(3, 16, 3, 2),  # Conv1: Input -> 16 channels
            Conv(16, 32, 3, 2),  # Conv2: 16 -> 32 channels
            C2f(32, 32, n=1),  # C2f1: Cross-Stage Partial (32 channels)
            Conv(32, 64, 3, 2),  # Conv3: 32 -> 64 channels
            C2f(64, 64, n=2),  # C2f2: Cross-Stage Partial (64 channels)
            Conv(64, 128, 3, 2),  # Conv4: 64 -> 128 channels
            C2f(128, 128, n=2),  # C2f3: Cross-Stage Partial (128 channels)
            Conv(128, 256, 3, 2),  # Conv5: 128 -> 256 channels
            C2f(256, 256, n=1),  # C2f4: Cross-Stage Partial (256 channels)
        )

        self.sppf = SPPF(256, 128)

        self.neck = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # Upsample1:
            nn.Identity(),  # Placeholder: Concat with backbone output
            C2f(384, 128, n=1),  # C2f5: Cross-Stage Partial (384 -> 128 channels)
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # Upsample2:
            nn.Identity(),  # Placeholder: Concat with previous neck output
            C2f(192, 64, n=1),  # C2f6: Cross-Stage Partial (192 -> 64 channels)
        )

        self.head = nn.Sequential(
            Conv(64, 64, 3, 2),  # Conv6: 64 -> 64 channels
            nn.Identity(),  # Placeholder: Concat with backbone output
            C2f(192, 128, n=1),  # C2f7: Cross-Stage Partial (192 -> 128 channels)
            Conv(128, 128, 3, 2),  # Conv7: 128 -> 128 channels
            nn.Identity(),  # Placeholder: Concat with previous neck output
            C2f(384, 256, n=1),  # C2f8: Cross-Stage Partial (384 -> 256 channels)
        )

        self.detect = Detect(num_classes, anchors=(), ch=(64, 128, 256))  # 檢測頭

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_backbone = self.backbone(x)
        x_sppf = self.sppf(x_backbone)
        x_neck = self.neck(x_sppf)
        x_head = self.head(x_neck)
        return self.detect([x_backbone, x_sppf, x_neck, x_head])


model = YOLOv8(num_classes=80)


def convert_annotations(json_path: str, output_dir: str) -> None:
    with open(json_path, "r") as f:
        annotations = json.load(f)

    for image in annotations:
        image_name = image["name"]
        width = image["width"]
        height = image["height"]
        annotations_list = image["annotations"]

        yolo_annotations = []
        for annotation in annotations_list:
            x_min = annotation["x"]
            y_min = annotation["y"]
            w = annotation["width"]
            h = annotation["height"]

            x_center = (x_min + w / 2) / width
            y_center = (y_min + h / 2) / height
            norm_width = w / width
            norm_height = h / height

            class_id = 0
            yolo_annotations.append(
                f"{class_id} {x_center} {y_center} {norm_width} {norm_height}"
            )

        output_txt_path = os.path.join(output_dir, image_name.replace(".jpg", ".txt"))
        with open(output_txt_path, "w") as out_file:
            out_file.write("\n".join(yolo_annotations))
