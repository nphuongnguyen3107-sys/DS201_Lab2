import torch
import torch.nn as nn


def n_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class BasicBlock(nn.Module):
    """
    Khối thặng dư cơ bản (BasicBlock) của ResNet-18/34.
    Bao gồm 2 lớp tích chập 3x3 kèm BatchNorm.
    Nếu kích thước hoặc số channel thay đổi (stride != 1 hoặc in_ch != out_ch),
    nhánh tắt (shortcut) sẽ dùng phép chiếu projection Conv 1x1 + BatchNorm.
    Cộng trước, ReLU sau: out = ReLU(F(x) + shortcut(x)).
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity                     # Cộng trước
        return torch.relu(out)              # ReLU sau


class ResNet18(nn.Module):
    """
    Kiến trúc ResNet-18 hoàn chỉnh gồm 4 stage [2, 2, 2, 2] BasicBlock.
    Khuyến nghị dùng ảnh đầu vào 96x96 hoặc 112x112 vì ResNet giảm chiều 32 lần (2^5).
    Tổng số tham số cho 21 lớp: 11,187,285 tham số.
    """
    def __init__(self, in_ch=3, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_stage(64, 64, stride=1)
        self.layer2 = self._make_stage(64, 128, stride=2)
        self.layer3 = self._make_stage(128, 256, stride=2)
        self.layer4 = self._make_stage(256, 512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, stride):
        # Block đầu tiên đổi channel và giảm kích thước không gian nếu stride=2
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == "__main__":
    ident = BasicBlock(64, 64, stride=1)
    proj = BasicBlock(64, 128, stride=2)
    assert n_params(ident) == 73984, f"Expected 73,984, got {n_params(ident)}"
    print("BasicBlock assertions passed!")

    r18 = ResNet18(num_classes=21)
    x = torch.rand(2, 3, 96, 96)
    out = r18(x)
    print("ResNet-18:")
    print("  Input shape :", tuple(x.shape))
    print("  Output shape:", tuple(out.shape))
    print("  Total params:", f"{n_params(r18):,}")
    assert n_params(r18) == 11187285, f"Expected 11,187,285, got {n_params(r18)}"
    print("  Assertion passed!")
